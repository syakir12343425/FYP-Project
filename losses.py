"""
Loss functions. Matches proposal Section 4.8, with the weight bug fixed:
lambda=0.01 (proposal) lets BCE dominate completely since BCE starts around
0.69 and MS-SSIM-loss starts near 0. config.yaml now defaults to 0.3 — tune
this with the sweep in train.py --sweep-lambda if PSNR/BER trade off badly.
"""
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim


def reconstruction_loss(recovered_probs: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy between recovered bit probabilities and the
    original message bits. (Section 4.8.1)"""
    return nn.functional.binary_cross_entropy(recovered_probs, message)


def perceptual_loss(cover: torch.Tensor, stego: torch.Tensor) -> torch.Tensor:
    """1 - MS-SSIM(cover, stego). (Section 4.8.2)
    pytorch_msssim expects images in [0, 1] and needs at least 160x160-ish
    input for the default 5-scale pyramid — fine for our 512 crop size, but
    will error on very small crops/debug images. Use ssim() instead of
    ms_ssim() if you ever crop below ~160px."""
    return 1.0 - ms_ssim(cover, stego, data_range=1.0, size_average=True)


def chroma_loss(cover: torch.Tensor, stego: torch.Tensor) -> torch.Tensor:
    """MSE of the colour (Cb, Cr) part of the change, BT.601 YCbCr, inputs in [0, 1].
    Added after the 3 Nov 2026 analysis: MS-SSIM is computed per RGB channel and barely
    penalises smooth colour shifts, so the first model hid the message almost entirely in
    colour (visible pink/green blotches, failures on very colourful photos). This term makes
    colour change expensive, pushing the message into brightness."""
    d = stego - cover
    r, g, b = d[:, 0], d[:, 1], d[:, 2]
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return (cb ** 2).mean() + (cr ** 2).mean()


def total_loss(
    cover: torch.Tensor,
    stego: torch.Tensor,
    message: torch.Tensor,
    recovered_probs: torch.Tensor,
    lambda_perceptual: float = 0.3,
    lambda_mse: float = 0.0,
    bce_weight: float = 1.0,
    lambda_chroma: float = 0.0,
):
    l_rec = reconstruction_loss(recovered_probs, message)
    l_per = perceptual_loss(cover, stego)
    l_mse = nn.functional.mse_loss(cover, stego) if lambda_mse > 0 else torch.tensor(0.0, device=cover.device)

    l_chroma = chroma_loss(cover, stego) if lambda_chroma > 0 else torch.tensor(0.0, device=cover.device)

    l_total = bce_weight * l_rec + lambda_perceptual * l_per + lambda_mse * l_mse + lambda_chroma * l_chroma
    return l_total, {
        "bce": l_rec.item(),
        "perceptual": l_per.item(),
        "mse": l_mse.item() if lambda_mse > 0 else 0.0,
        "chroma": l_chroma.item() if lambda_chroma > 0 else 0.0,
        "total": l_total.item(),
    }
