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
