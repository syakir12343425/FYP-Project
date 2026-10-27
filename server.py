"""
Inference server for the FYP steganography frontend.

Loads the trained CNN (checkpoints/best.pt) once at startup and exposes three
endpoints that map directly onto the four pages of index.html:

    POST /api/embed       cover image + message + method -> stego image + psnr/ssim
    POST /api/downscale   image + algorithm + scale        -> downscaled image
    POST /api/extract     image + method + original message -> recovered message + ber

Run it from inside the fyp_stego/ folder (so `models.py`, `downscale.py`,
`lsb_baseline.py` are importable, and `checkpoints/best.pt` exists next to it):

    pip install fastapi uvicorn pillow scikit-image python-multipart opencv-python-headless
    uvicorn server:app --reload --host 0.0.0.0 --port 8000

Then point the frontend's fetch() calls at http://<this machine's address>:8000/api/...
"""
import base64
import io

import cv2
import numpy as np
import torch
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1280 * 1280
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models import Encoder, Decoder
from downscale import cv2_downscale_numpy
from lsb_baseline import lsb_embed, lsb_extract

import config
from auth_routes import limiter, require_user, router as auth_router
from db import AdminUser, SessionLocal, audit, get_db, init_db

MESSAGE_BITS = 1024          # must match training (configs/config.yaml -> message.bits)
CHECKPOINT_PATH = "checkpoints/best.pt"   # copy best.pt down from Drive to here
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

is_prod = config.ENV == "prod"
app = FastAPI(
    title="FYP Steganography Inference Server",
    docs_url=None if is_prod else "/docs", redoc_url=None, openapi_url=None if is_prod else "/openapi.json",
)
init_db()

# --- CORS: explicit origins only (never "*" together with credentials) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)

# --- rate limiting (login is 5/min/IP; see auth_routes) + default for everything else ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

MAX_BODY_BYTES = 12 * 1024 * 1024   # largest legit request = two ~512px PNG data URLs


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Reject oversized bodies early (Content-Length) to blunt memory-exhaustion DoS.
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
        return JSONResponse({"detail": "Payload too large"}, status_code=413)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    if config.COOKIE_SECURE:
        resp.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return resp


app.include_router(auth_router)

# ---------------------------------------------------------------- model load
encoder = Encoder(message_bits=MESSAGE_BITS).to(DEVICE)
decoder = Decoder(message_bits=MESSAGE_BITS).to(DEVICE)

_ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
encoder.load_state_dict(_ckpt["encoder"])
decoder.load_state_dict(_ckpt["decoder"])
encoder.eval()
decoder.eval()
print(
    f"Loaded {CHECKPOINT_PATH} (epoch {_ckpt.get('epoch')}, "
    f"val_ber={_ckpt.get('val_ber')}, val_psnr={_ckpt.get('val_psnr')}) on {DEVICE}"
)


# --------------------------------------------------------------- image utils
def decode_base64_image_raw(data_url: str) -> np.ndarray:
    """Accepts a canvas.toDataURL() string ('data:image/png;base64,...') or
    raw base64. Returns an HxWx3 uint8 RGB numpy array AT ITS ACTUAL SIZE,
    no resizing. Use this for anything that must keep its real resolution
    (the downscaled/attacked image going into the decoder) — the decoder
    is fully convolutional (AdaptiveAvgPool2d) specifically so it can accept
    any input size, which is the whole point of testing robustness at
    different resolutions."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url, validate=True)
        img = Image.open(io.BytesIO(raw))
        if img.format not in ("PNG", "JPEG"):
            raise ValueError("unsupported format")
        w, h = img.size            # header only, nothing decoded yet -> blocks decompression bombs
        if not (32 <= w <= 1280 and 32 <= h <= 1280):
            raise ValueError("image dimensions out of range")
        arr = np.array(img.convert("RGB"))
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid image payload")
    return arr


def decode_base64_image_512(data_url: str) -> np.ndarray:
    """Same as above, but forces the result to 512x512 — use this ONLY for
    the cover image going into the encoder, since training used 512x512
    crops. Never use this on an already-downscaled/attacked image, or
    you'd silently undo the attack before the decoder ever sees it."""
    arr = decode_base64_image_raw(data_url)
    if arr.shape[0] != 512 or arr.shape[1] != 512:
        img = Image.fromarray(arr).resize((512, 512), Image.LANCZOS)
        arr = np.array(img)
    return arr


def encode_image_base64(arr: np.ndarray) -> str:
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def to_tensor(img_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1)


def to_uint8_array(t: torch.Tensor) -> np.ndarray:
    arr = t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0
    return arr.round().astype(np.uint8)


# ------------------------------------------------------------- message codec
def text_to_bits(text: str, n_bits: int = MESSAGE_BITS) -> np.ndarray:
    n_bytes = n_bits // 8
    data = text.encode("utf-8")[:n_bytes].ljust(n_bytes, b"\x00")
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_text(bits: np.ndarray) -> str:
    byte_arr = np.packbits(np.asarray(bits, dtype=np.uint8))
    raw = byte_arr.tobytes().split(b"\x00", 1)[0]  # stop at the zero-padding
    return raw.decode("utf-8", errors="replace")


# ------------------------------------------------------------------ schemas
MAX_IMG_CHARS = 5_500_000   # ~4 MB of base64

class EmbedRequest(BaseModel):
    cover_image: str = Field(max_length=MAX_IMG_CHARS)
    message: str = Field(min_length=1, max_length=120)
    method: Literal["cnn", "lsb"]


class EmbedResponse(BaseModel):
    stego_image: str
    psnr: float
    ssim: float


class DownscaleRequest(BaseModel):
    image: str = Field(max_length=MAX_IMG_CHARS)
    algorithm: Literal["nearest", "bilinear", "bicubic", "lanczos", "area"]
    scale: float = Field(ge=0.43, le=0.76)   # only the benchmarked range (224..384 px of 512)


class DownscaleResponse(BaseModel):
    downscaled_image: str
    size: int


class ExtractRequest(BaseModel):
    image: str = Field(max_length=MAX_IMG_CHARS)
    method: Literal["cnn", "lsb"]
    original_message: str = Field(default="", max_length=120)
    cover_image: str = Field(default="", max_length=MAX_IMG_CHARS)


class ExtractResponse(BaseModel):
    recovered_message: str
    ber: float | None = None
    extracted_psnr: float | None = None   # cover vs. (downscaled image upscaled back to 512x512)
    extracted_ssim: float | None = None


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/embed", response_model=EmbedResponse)
@limiter.limit("30/minute")
def embed(request: Request, req: EmbedRequest, user: AdminUser = Depends(require_user),
          db: Session = Depends(get_db)):
    cover_np = decode_base64_image_512(req.cover_image)
    bits = text_to_bits(req.message)

    if req.method == "cnn":
        cover_t = to_tensor(cover_np).unsqueeze(0).to(DEVICE)
        msg_t = torch.from_numpy(bits).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            stego_t = encoder(cover_t, msg_t)
        stego_np = to_uint8_array(stego_t.squeeze(0))
    else:  # lsb
        stego_np = lsb_embed(cover_np, bits)

    psnr = float(peak_signal_noise_ratio(cover_np, stego_np, data_range=255))
    ssim = float(structural_similarity(cover_np, stego_np, channel_axis=2, data_range=255))

    audit(db, "embed", request, user.username, f"method={req.method} msg_len={len(req.message)} psnr={psnr:.2f}")
    return EmbedResponse(stego_image=encode_image_base64(stego_np), psnr=psnr, ssim=ssim)


@app.post("/api/downscale", response_model=DownscaleResponse)
@limiter.limit("60/minute")
def downscale(request: Request, req: DownscaleRequest, user: AdminUser = Depends(require_user),
              db: Session = Depends(get_db)):
    img_np = decode_base64_image_raw(req.image)
    orig_size = img_np.shape[0]
    target_size = max(1, round(orig_size * req.scale))
    out_np = cv2_downscale_numpy(img_np, target_size, req.algorithm)
    audit(db, "downscale", request, user.username, f"algo={req.algorithm} scale={req.scale} size={target_size}")
    return DownscaleResponse(downscaled_image=encode_image_base64(out_np), size=target_size)


@app.post("/api/extract", response_model=ExtractResponse)
@limiter.limit("60/minute")
def extract(request: Request, req: ExtractRequest, user: AdminUser = Depends(require_user),
            db: Session = Depends(get_db)):
    # IMPORTANT: raw, not resized — the decoder must see the image at its
    # actual (post-attack) resolution, exactly like eval.py does, otherwise
    # you're not really testing robustness to downscaling at all.
    img_np = decode_base64_image_raw(req.image)

    if req.method == "cnn":
        img_t = to_tensor(img_np).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = decoder(img_t)
        bits = (probs.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
    else:  # lsb
        bits = lsb_extract(img_np, MESSAGE_BITS)

    recovered_text = bits_to_text(bits)

    ber = None
    if req.original_message:
        true_bits = text_to_bits(req.original_message)
        ber = float(np.mean(true_bits != bits))

    extracted_psnr = None
    extracted_ssim = None
    if req.cover_image:
        cover_np = decode_base64_image_512(req.cover_image)
        # Upscale the (possibly smaller) attacked image back to the cover's
        # size PURELY for this quality comparison — this upscaled copy is
        # never what the decoder saw above. This mirrors how a robustness
        # paper reports "quality after a resize round-trip".
        if img_np.shape[:2] != cover_np.shape[:2]:
            roundtrip_np = cv2.resize(
                img_np, (cover_np.shape[1], cover_np.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        else:
            roundtrip_np = img_np
        extracted_psnr = float(peak_signal_noise_ratio(cover_np, roundtrip_np, data_range=255))
        extracted_ssim = float(
            structural_similarity(cover_np, roundtrip_np, channel_axis=2, data_range=255)
        )

    audit(db, "extract", request, user.username,
          f"method={req.method} size={img_np.shape[0]} ber={'' if ber is None else round(ber, 4)}")
    return ExtractResponse(
        recovered_message=recovered_text,
        ber=ber,
        extracted_psnr=extracted_psnr,
        extracted_ssim=extracted_ssim,
    )
