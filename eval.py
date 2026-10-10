"""
Full evaluation: runs the proposal's Experiment 2 / 3 / 4 matrix —
100 test images x 5 algorithms x 4 target sizes, for both the trained CNN
and the LSB baseline — and writes results to a CSV (matches Section 4.10.4).

Usage:
    python eval.py --config configs/config.yaml --checkpoint checkpoints/best.pt

Outputs:
    results/eval_results.csv       one row per (image, method, algorithm, size)
    results/eval_summary.csv       mean/std per (method, algorithm, size)
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from tqdm import tqdm

from dataset import DIV2KStegoDataset
from downscale import cv2_downscale_numpy
from lsb_baseline import lsb_embed, lsb_extract
from models import Decoder, Encoder


def to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """(3,H,W) float [0,1] -> HxWx3 uint8"""
    arr = (img_tensor.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    message_bits = ckpt["config"]["message"]["bits"]

    encoder = Encoder(message_bits=message_bits, base_channels=cfg["model"]["encoder_channels"],
                       residual_scale=cfg["model"]["residual_scale"]).to(device)
    decoder = Decoder(message_bits=message_bits, base_channels=cfg["model"]["decoder_channels"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, payload={message_bits} bits")

    test_ds = DIV2KStegoDataset(
        cfg["data"]["val_dir"], crop_size=cfg["data"]["crop_size"],
        message_bits=message_bits, train=False,
    )
    n_test = min(cfg["eval"]["n_test_images"], len(test_ds))
    print(f"Evaluating on {n_test} images x {len(cfg['eval']['algorithms'])} algorithms "
          f"x {len(cfg['eval']['target_sizes'])} sizes = "
          f"{n_test * len(cfg['eval']['algorithms']) * len(cfg['eval']['target_sizes'])} combinations per method")

    rows = []
    rng = np.random.default_rng(42)

    with torch.no_grad():
        for idx in tqdm(range(n_test)):
            cover_t, message_t = test_ds[idx]
            cover_t = cover_t.unsqueeze(0).to(device)
            message_t = message_t.unsqueeze(0).to(device)

            # --- CNN embed (once per image) ---
            stego_t = encoder(cover_t, message_t)
            cover_np = to_uint8(cover_t[0])
            stego_np_cnn = to_uint8(stego_t[0])
            embed_psnr_cnn = sk_psnr(cover_np, stego_np_cnn, data_range=255)
            embed_ssim_cnn = sk_ssim(cover_np, stego_np_cnn, channel_axis=2, data_range=255)

            # --- LSB embed (once per image, same payload size) ---
            message_np = message_t[0].cpu().numpy().astype(np.uint8)
            stego_np_lsb = lsb_embed(cover_np, message_np)
            embed_psnr_lsb = sk_psnr(cover_np, stego_np_lsb, data_range=255)
            embed_ssim_lsb = sk_ssim(cover_np, stego_np_lsb, channel_axis=2, data_range=255)

            for algo in cfg["eval"]["algorithms"]:
                for size in cfg["eval"]["target_sizes"]:
                    # ---- CNN path ----
                    down_cnn = cv2_downscale_numpy(stego_np_cnn, size, algo)
                    down_cnn_t = torch.from_numpy(down_cnn).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                    recovered = decoder(down_cnn_t)
                    pred_bits = (recovered[0] > 0.5).float().cpu().numpy().astype(np.uint8)
                    ber_cnn = float(np.mean(pred_bits != message_np))

                    rows.append({
                        "image_idx": idx, "method": "cnn", "algorithm": algo, "size": size,
                        "ber": ber_cnn, "embed_psnr": embed_psnr_cnn, "embed_ssim": embed_ssim_cnn,
                    })

                    # ---- LSB path ----
                    down_lsb = cv2_downscale_numpy(stego_np_lsb, size, algo)
                    # LSB "decode": resize attacked image back up to original res (closest
                    # fair comparison when no training-time size target exists for LSB),
                    # then read bits directly — this is intentionally a best case for LSB.
                    up_lsb = cv2_downscale_numpy(down_lsb, cover_np.shape[0], algo)
                    extracted = lsb_extract(up_lsb, message_bits)
                    ber_lsb = float(np.mean(extracted != message_np))

                    rows.append({
                        "image_idx": idx, "method": "lsb", "algorithm": algo, "size": size,
                        "ber": ber_lsb, "embed_psnr": embed_psnr_lsb, "embed_ssim": embed_ssim_lsb,
                    })

    # --- write raw results ---
    csv_path = out_dir / "eval_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")

    # --- summary ---
    import pandas as pd
    df = pd.DataFrame(rows)
    summary = df.groupby(["method", "algorithm", "size"]).agg(
        mean_ber=("ber", "mean"), std_ber=("ber", "std"),
        mean_psnr=("embed_psnr", "mean"), mean_ssim=("embed_ssim", "mean"),
    ).reset_index()
    summary_path = out_dir / "eval_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
