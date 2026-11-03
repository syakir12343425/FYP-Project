"""
Differentiable downscaling module.

Implements the "attack layer" from Chapter 4 of the proposal. Unlike the
proposal's claim that only bilinear/bicubic are differentiable, ALL standard
resizing operations are linear in the input pixels (Id = sum_i w_i * Is_i),
so gradients exist for all of them. What actually differs is whether a
*library* ships an autograd-friendly implementation:

  - bilinear, bicubic, nearest, area -> torch.nn.functional.interpolate
    (all differentiable in PyTorch directly)
  - lanczos -> not implemented in F.interpolate. We either (a) hold it out
    as a zero-shot test-only algorithm (recommended, see config
    downscale.held_out_mode), or (b) implement a fixed-kernel conv version
    (see LanczosDownscale below) if you want it in training too.

Train-time usage: randomly sample a target size and mode per batch, so the
encoder/decoder see varied downscaling conditions (matches config.yaml
downscale.train_sizes / train_modes).

Test-time usage: run every (algorithm, size) combination explicitly for the
evaluation matrix (eval.py).
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch interpolate modes that are differentiable out of the box
_TORCH_MODES = {
    "nearest": "nearest",
    "bilinear": "bilinear",
    "bicubic": "bicubic",
    "area": "area",
}


def quantize_ste(x: torch.Tensor) -> torch.Tensor:
    """
    Straight-through 8-bit quantization. Simulates the fact that real images
    are saved/loaded as uint8 (0-255) at some point in an AI pipeline, which
    the proposal's pipeline never modeled. Forward pass actually quantizes;
    backward pass passes the gradient through unchanged (STE), so the
    encoder still gets a training signal despite the non-differentiable
    round().
    """
    x_clamped = x.clamp(0, 1)
    quantized = torch.round(x_clamped * 255.0) / 255.0
    # straight-through: forward = quantized, backward = identity
    return x_clamped + (quantized - x_clamped).detach()


def _codec_roundtrip_uint8(img_hwc_uint8, fmt: str, quality: int):
    """Real encode/decode with Pillow (no autograd). JPEG uses 4:2:0 chroma subsampling, like
    messaging apps; WebP lossy is 4:2:0 by design. Returns the decoded HxWx3 uint8 array."""
    import io

    import numpy as np
    from PIL import Image
    buf = io.BytesIO()
    if fmt == "jpeg":
        Image.fromarray(img_hwc_uint8).save(buf, "JPEG", quality=int(quality), subsampling="4:2:0")
    elif fmt == "webp":
        Image.fromarray(img_hwc_uint8).save(buf, "WEBP", quality=int(quality))
    else:
        raise ValueError(fmt)
    return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def codec_ste(x: torch.Tensor, fmt: str, quality: int) -> torch.Tensor:
    """
    Straight-through JPEG/WebP (added 3 Nov 2026 for the WhatsApp experiment). Forward = a REAL
    Pillow encode/decode of each image at the given quality, so the decoder sees true codec
    artefacts (8x8 DCT quantisation, reduced colour resolution). Backward = identity, the same
    trick as quantize_ste, so the encoder still gets a gradient.
    """
    import numpy as np
    xc = x.float().clamp(0, 1)
    with torch.no_grad():
        arr = (xc.detach().permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        out = np.stack([_codec_roundtrip_uint8(a, fmt, quality) for a in arr])
        y = torch.from_numpy(out).to(xc.device).permute(0, 3, 1, 2).float() / 255.0
    return xc + (y - xc).detach()


def lanczos_kernel(x: torch.Tensor, a: int = 3) -> torch.Tensor:
    pi = torch.pi
    x = x.clone()
    out = torch.where(
        x.abs() < 1e-8,
        torch.ones_like(x),
        a * torch.sin(pi * x) * torch.sin(pi * x / a) / (pi ** 2 * x ** 2 + 1e-8),
    )
    out = torch.where(x.abs() >= a, torch.zeros_like(out), out)
    return out


class DifferentiableDownscale(nn.Module):
    """
    Wraps F.interpolate for nearest/bilinear/bicubic/area (all differentiable),
    plus optional 8-bit quantization before and after resizing to simulate a
    realistic save/load step in an AI pipeline.

    For Lanczos we do NOT provide a differentiable path by default — it is
    used as a held-out, test-only algorithm for the cross-algorithm
    generalization experiment (Experiment 4 / 3.5.5 in the proposal), which
    is a cleaner and more defensible setup than forcing a from-scratch
    autograd Lanczos kernel into training.
    """

    def __init__(self, quantize: bool = True):
        super().__init__()
        self.quantize = quantize

    def forward(self, x: torch.Tensor, size: int, mode: str) -> torch.Tensor:
        assert mode in _TORCH_MODES, (
            f"'{mode}' has no differentiable torch path; use this module only "
            f"for {list(_TORCH_MODES)}. Lanczos is test-only — see eval.py."
        )
        if self.quantize:
            x = quantize_ste(x)

        kwargs = {}
        if mode in ("bilinear", "bicubic"):
            kwargs["align_corners"] = False
        out = F.interpolate(x, size=(size, size), mode=_TORCH_MODES[mode], **kwargs)
        out = out.clamp(0, 1)

        if self.quantize:
            out = quantize_ste(out)
        return out

    def random_forward(self, x: torch.Tensor, sizes: list, modes: list):
        """Sample one (size, mode) per call — used inside the training loop."""
        size = random.choice(sizes)
        mode = random.choice(modes)
        return self.forward(x, size, mode), size, mode

    def channel_forward(self, x: torch.Tensor, attacks: list, jpeg_quality: list, webp_quality: list):
        """
        Messaging-app attacks at FULL size (no resize), added 3 Nov 2026: the first models were only
        ever trained on 224-384 px resizes, so a 512 px sticker or photo was an untrained case.
          none -> 8-bit save/load only (a lossless sticker or document)
          jpeg -> real JPEG at a random quality in jpeg_quality [lo, hi] (WhatsApp photo mode)
          webp -> real lossy WebP at a random quality in webp_quality [lo, hi] (WhatsApp sticker)
        Returns (image, size, label) like random_forward.
        """
        attack = random.choice(attacks)
        out = quantize_ste(x) if self.quantize else x.clamp(0, 1)
        if attack == "jpeg":
            q = random.randint(*jpeg_quality)
            out = codec_ste(out, "jpeg", q)
            return out, x.shape[-1], f"jpeg{q}"
        if attack == "webp":
            q = random.randint(*webp_quality)
            out = codec_ste(out, "webp", q)
            return out, x.shape[-1], f"webp{q}"
        if attack != "none":
            raise ValueError(attack)
        return out, x.shape[-1], "none"


def cv2_downscale_numpy(img_uint8, size: int, algorithm: str):
    """
    Non-differentiable, test-time-only resize using OpenCV, for evaluating
    against the EXACT algorithms real AI pipelines use (including lanczos,
    which has no torch path here). Use this in eval.py, never inside the
    training loop.

    img_uint8: HxWx3 uint8 numpy array (RGB)
    Returns: size x size x 3 uint8 numpy array
    """
    import cv2

    codes = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
    }
    assert algorithm in codes, f"unknown algorithm {algorithm}"
    return cv2.resize(img_uint8, (size, size), interpolation=codes[algorithm])
