"""Does the CNN's BER depend on WHAT the 1024 bits look like?

eval.py (and training) only ever used uniformly random bits. Real messages are text: regular
ASCII patterns followed by a long run of zero padding. This script embeds three payloads into the
SAME cover, applies the SAME OpenCV attack, and reports BER for each:

  random     - uniformly random bits (what the benchmark measured)
  text       - your message + zero padding, exactly as server.py sends it today
  whitened   - the same text through payload.py (what server.py embeds now)

Usage (repo root, needs checkpoints/best.pt; runs on CPU in a few seconds per image):
    python tools/payload_experiment.py path\\to\\photo.png
    python tools/payload_experiment.py path\\to\\photo.png --algo lanczos --size 384 --msg "hello"
Several images: pass several paths. Results are printed only; nothing is written.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from downscale import cv2_downscale_numpy   # noqa: E402
from models import Decoder, Encoder          # noqa: E402
from payload import decode_bits, encode_message  # noqa: E402  (the codec server.py uses)

BITS = 1024


def load_cover(path: str) -> np.ndarray:
    """Same as server.py: force the cover to 512x512 RGB."""
    img = Image.open(path).convert("RGB")
    if img.size != (512, 512):
        img = img.resize((512, 512), Image.LANCZOS)
    return np.array(img)


def text_bits(msg: str) -> np.ndarray:
    data = msg.encode("utf-8")[: BITS // 8].ljust(BITS // 8, b"\x00")
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def run(encoder, decoder, cover_np, bits, algo, size):
    cover = torch.from_numpy(cover_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    msg = torch.from_numpy(bits.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        stego = encoder(cover, msg)
    stego_np = (stego.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    attacked = cv2_downscale_numpy(stego_np, size, algo)
    att = torch.from_numpy(attacked.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        out = (decoder(att)[0].numpy() > 0.5).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--algo", default="bicubic", choices=["nearest", "bilinear", "bicubic", "lanczos", "area"])
    ap.add_argument("--size", type=int, default=384, choices=[224, 256, 299, 384])
    ap.add_argument("--msg", default="try something lagi la kawan")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    encoder, decoder = Encoder(message_bits=BITS), Decoder(message_bits=BITS)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    tb = text_bits(args.msg)
    n_msg = len(args.msg.encode("utf-8")[: BITS // 8]) * 8
    rand = np.random.default_rng(0).integers(0, 2, BITS, dtype=np.uint8)

    print(f"attack: {args.algo} -> {args.size}px | message: {n_msg // 8} bytes + {BITS - n_msg} padding bits\n")
    print(f"{'image':28s} {'random':>8s} {'text':>8s} {'  (msg part / padding)':>24s} {'whitened':>9s}")
    for path in args.images:
        cover = load_cover(path)
        r = run(encoder, decoder, cover, rand, args.algo, args.size)
        t = run(encoder, decoder, cover, tb, args.algo, args.size)
        w = decode_bits(run(encoder, decoder, cover, encode_message(args.msg), args.algo, args.size))
        ber = lambda got, ref: 100 * float(np.mean(got != ref))
        msg_part = ber(t[:n_msg], tb[:n_msg])
        pad_part = ber(t[n_msg:], tb[n_msg:])
        print(f"{Path(path).name[:28]:28s} {ber(r, rand):7.2f}% {ber(t, tb):7.2f}% "
              f"{f'({msg_part:.1f}% / {pad_part:.1f}%)':>24s} {ber(w, tb):8.2f}%")


if __name__ == "__main__":
    main()
