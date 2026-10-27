"""
End-to-end training loop (Algorithm 1 in the proposal, corrected).

Usage (Colab, GPU runtime on):
    python train.py --config configs/config.yaml

Resume:
    python train.py --config configs/config.yaml --resume checkpoints/last.pt
"""
import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DIV2KStegoDataset
from downscale import DifferentiableDownscale
from losses import total_loss
from models import Decoder, Encoder, count_parameters


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def bit_error_rate(message: torch.Tensor, recovered_probs: torch.Tensor) -> float:
    pred_bits = (recovered_probs > 0.5).float()
    return (pred_bits != message).float().mean().item()


def evaluate_quick(encoder, decoder, downscaler, loader, device, cfg, n_batches=10):
    """Fast validation pass: fixed set of (size, mode) sampled once per batch,
    averaged BER/PSNR over a few batches. Full eval.py does the complete
    5-algorithm x 4-size matrix — this is just for picking the best checkpoint
    during training."""
    encoder.eval()
    decoder.eval()
    total_ber, total_psnr, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i, (cover, message) in enumerate(loader):
            if i >= n_batches:
                break
            cover, message = cover.to(device), message.to(device)
            stego = encoder(cover, message)
            downscaled, size, mode = downscaler.random_forward(
                stego, cfg["downscale"]["train_sizes"], cfg["downscale"]["train_modes"]
            )
            recovered = decoder(downscaled)
            total_ber += bit_error_rate(message, recovered)

            mse = torch.nn.functional.mse_loss(cover, stego).item()
            psnr = 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-10))).item()
            total_psnr += psnr
            n += 1
    encoder.train()
    decoder.train()
    return total_ber / max(n, 1), total_psnr / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--message-bits", type=int, default=None, help="override config.message.bits (for ablation runs)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU detected. This will be extremely slow — "
              "run this on Colab/Kaggle with a GPU runtime, not locally.")

    message_bits = args.message_bits or cfg["message"]["bits"]
    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    train_ds = DIV2KStegoDataset(
        cfg["data"]["train_dir"], crop_size=cfg["data"]["crop_size"],
        message_bits=message_bits, train=True,
    )
    val_ds = DIV2KStegoDataset(
        cfg["data"]["val_dir"], crop_size=cfg["data"]["crop_size"],
        message_bits=message_bits, train=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)} | payload: {message_bits} bits")

    # --- model ---
    encoder = Encoder(
        message_bits=message_bits,
        base_channels=cfg["model"]["encoder_channels"],
        residual_scale=cfg["model"]["residual_scale"],
    ).to(device)
    decoder = Decoder(
        message_bits=message_bits,
        base_channels=cfg["model"]["decoder_channels"],
    ).to(device)
    downscaler = DifferentiableDownscale(quantize=True).to(device)

    print(f"Encoder params: {count_parameters(encoder):,} | Decoder params: {count_parameters(decoder):,}")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg["train"]["lr"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")

    start_epoch = 0
    best_ber = 1.0
    patience_counter = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_ber = ckpt.get("best_ber", 1.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        t0 = time.time()
        running = {"bce": 0.0, "perceptual": 0.0, "total": 0.0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")

        for step, (cover, message) in enumerate(pbar):
            cover, message = cover.to(device), message.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=cfg["train"]["amp"] and device.type == "cuda"):
                stego = encoder(cover, message)
                downscaled, size, mode = downscaler.random_forward(
                    stego, cfg["downscale"]["train_sizes"], cfg["downscale"]["train_modes"]
                )
                recovered = decoder(downscaled)

            # Loss is computed OUTSIDE autocast, in float32: BCE-after-sigmoid is unsafe under
            # autocast (raises RuntimeError), and MS-SSIM is more stable in fp32 anyway.
            loss, parts = total_loss(
                cover.float(), stego.float(), message.float(), recovered.float(),
                lambda_perceptual=cfg["loss"]["lambda_perceptual"],
                lambda_mse=cfg["loss"]["lambda_mse"],
                bce_weight=cfg["loss"]["bce_weight"],
            )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            for k in running:
                running[k] += parts[k]

            if step % cfg["train"]["log_every"] == 0:
                pbar.set_postfix({k: f"{v/(step+1):.4f}" for k, v in running.items()})

        val_ber, val_psnr = evaluate_quick(encoder, decoder, downscaler, val_loader, device, cfg)
        dt = time.time() - t0
        print(f"Epoch {epoch+1}: val BER={val_ber:.4f} val PSNR={val_psnr:.2f}dB ({dt:.0f}s)")

        ckpt = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_ber": val_ber,
            "val_psnr": val_psnr,
            "best_ber": best_ber,
            "config": cfg,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        if val_ber < best_ber:
            best_ber = val_ber
            patience_counter = 0
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  -> new best (BER={best_ber:.4f}), saved checkpoints/best.pt")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stop_patience"]:
                print(f"Early stopping: no improvement for {patience_counter} epochs.")
                break

    print("Training complete. Run eval.py on checkpoints/best.pt for the full benchmark matrix.")


if __name__ == "__main__":
    main()
