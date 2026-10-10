# FYP: Robust Deep Image Steganography Against Downscaling

PyTorch implementation, built from the redesign of the Chapter 4 proposal
(fixes payload mismatch, deepens the networks, corrects loss weighting,
trains on all differentiable resize algorithms).

## Files
| File | Purpose |
|---|---|
| `configs/config.yaml` | All hyperparameters — edit this, not the code |
| `downscale.py` | Differentiable resize layer (the "attack" simulator) |
| `models.py` | `Encoder` and `Decoder` CNNs |
| `losses.py` | BCE + MS-SSIM loss |
| `dataset.py` | DIV2K loader with random-crop (not resize) |
| `lsb_baseline.py` | LSB embed/extract, same payload, for the comparison |
| `train.py` | Training loop, checkpointing, early stopping |
| `eval.py` | Full 5-algorithm x 4-size benchmark matrix vs LSB |
| `colab_starter.ipynb` | Paste-and-run notebook for Colab |

## Quick start (Colab)
1. Push this folder to a GitHub repo.
2. Open `colab_starter.ipynb` in Colab.
3. Runtime -> Change runtime type -> T4 GPU.
4. Run cells top to bottom.

## Quick start (local, code-testing only — not training)
```bash
pip install -r requirements.txt
python models.py          # shape sanity check, runs on CPU in seconds
```
Do not try to run `train.py` locally on a Ryzen/Radeon laptop — `torch.cuda.is_available()`
will be `False` and training will be unusably slow. Write/debug code locally, train on Colab.

## What changed vs. the original proposal, and why

1. **Payload fixed at 1024 bits** (32x32 grid), matching exactly what the
   decoder's `AdaptiveAvgPool2d` can output. The proposal's "0.5 bpp =
   1,572,864 bits" was inconsistent with its own decoder design.
2. **Six conv layers instead of four**, with BatchNorm, for a larger
   receptive field relative to the >=4x downscale ratio.
3. **`residual_scale=0.03`** instead of `0.1` — the proposal's 0.1 caps PSNR
   near ~20dB; 0.03 targets the 35-38dB stated elsewhere in the proposal.
4. **`lambda_perceptual=0.3`** instead of `0.01` — at 0.01 the perceptual
   loss is negligible next to BCE, making Table 4.7 in the proposal
   untested in practice.
5. **Trains on nearest/bilinear/bicubic/area** (all differentiable via
   `F.interpolate`), not just bilinear/bicubic — the proposal's claim that
   only two algorithms are differentiable doesn't hold; resizing is linear
   in the input regardless of algorithm. Lanczos is held out as a
   zero-shot test-only algorithm for the generalization experiment
   (`downscale.held_out_mode` in config).
6. **8-bit quantization (straight-through estimator)** before/after
   resizing, simulating a real save/load step the original pipeline never
   modeled.
7. **Random crop instead of resize-to-1024x1024** for DIV2K preprocessing,
   avoiding aspect-ratio distortion. Default crop is 512x512 to fit a T4;
   bump to 1024 in config if you have more VRAM or lower the batch size.
8. **Mixed precision (AMP)** on by default — needed to fit batch_size=8 on
   a T4 at 512x512 with BatchNorm layers.

## Still to decide / verify once you have real numbers
- Run `train.py --sweep-lambda` style experiments (not yet scripted — add
  a loop over `loss.lambda_perceptual` values and compare `eval.py` output)
  to justify whatever lambda you report, since the proposal's Table 4.7 had
  no experiment behind it.
- Decide your one consistent set of success thresholds (PSNR, SSIM, BER)
  and use the same numbers in the proposal, the UI, and the report —
  right now they differ across chapters.
- Add Baseline 2 (same CNN, no downscale layer during training) to isolate
  what the differentiable downscale layer actually buys you — currently
  only the LSB baseline is implemented.
