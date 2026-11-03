"""Benchmark every embedding method under the same downscaling attacks (supervisor request:
compare several embedding algorithms, not only the CNN).

Methods: lsb, dct, dwt (numpy only) and cnn (needs torch + checkpoints/best.pt).
Every method embeds the SAME payload the server embeds (payload.py: Reed-Solomon + whitening, 1024 bits),
into the same 512x512 cover, and is attacked with the same OpenCV resize as server.py.

Per method it reports:
  embedding PSNR / SSIM (cover vs stego, before any attack)
  raw BER after each algorithm x size        (comparable with eval.py)
  exact recovery rate after Reed-Solomon      (what a user actually experiences)

Usage (repo root):
    python tools/benchmark_methods.py --images data/DIV2K_valid_HR --limit 100        # full benchmark
    python tools/benchmark_methods.py --images "C:\\Users\\me\\Pictures" --limit 10
    python tools/benchmark_methods.py --samples                                      # scikit-image photos, no download
    python tools/benchmark_methods.py --samples --methods lsb dct dwt                # skip the CNN
    python tools/benchmark_methods.py --samples --methods cnn --cnn-scale 0.015      # weaker CNN residual (no retraining)
    python tools/benchmark_methods.py --samples --methods cnn --cnn-luma             # keep only the brightness part of the CNN change
Writes results/methods_results.csv (one row per image x method x attack) and prints summary tables.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import classical_stego as cs                       # noqa: E402
from lsb_baseline import lsb_embed, lsb_extract    # noqa: E402
from payload import decode_payload, encode_message  # noqa: E402

ALGOS = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC,
         "lanczos": cv2.INTER_LANCZOS4, "area": cv2.INTER_AREA}
SIZES = [384, 299, 256, 224]
MESSAGE = "Hidden message that survives downscaling."


def square_512(img: np.ndarray) -> np.ndarray:
    """Centre-crop to a square, then resize to 512 (same cover size as the server)."""
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    return np.array(Image.fromarray(img[..., :3]).resize((512, 512), Image.LANCZOS))


def load_images(args):
    if args.samples:
        import skimage.data as d
        names = ["astronaut", "coffee", "chelsea", "rocket", "hubble_deep_field", "immunohistochemistry",
                 "retina", "camera", "brick", "grass", "gravel", "moon"]
        return [(n, square_512(getattr(d, n)())) for n in names][: args.limit]
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    return [(p.name, square_512(np.array(Image.open(p).convert("RGB")))) for p in paths[: args.limit]]


class CNN:
    """Thin wrapper so the CNN looks like the other methods (embed/extract on uint8 arrays)."""

    def __init__(self, checkpoint, residual_scale=None, luma_only=False):
        import torch
        from models import Decoder, Encoder
        self.torch = torch
        ck = torch.load(checkpoint, map_location="cpu")
        luma = ck.get("config", {}).get("model", {}).get("luma_only", False)   # brightness-only model?
        self.enc, self.dec = Encoder(message_bits=1024, luma_only=luma), Decoder(message_bits=1024)
        self.enc.load_state_dict(ck["encoder"])
        self.dec.load_state_dict(ck["decoder"])
        if residual_scale is not None:   # stego = cover + net(x) * residual_scale, so this scales the change linearly
            self.enc.residual_scale = residual_scale
        self.luma_only = luma_only
        self.enc.eval()
        self.dec.eval()

    def _t(self, a):
        return self.torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    def embed(self, cover, bits):
        with self.torch.no_grad():
            c = self._t(cover)
            s = self.enc(c, self.torch.from_numpy(bits.astype(np.float32)).unsqueeze(0))
            if self.luma_only:   # experiment: drop the colour part of the change, keep its brightness (BT.601 luma)
                d = s - c
                y = 0.299 * d[:, 0:1] + 0.587 * d[:, 1:2] + 0.114 * d[:, 2:3]
                s = c + y.expand_as(c)   # same amount added to R, G and B = brightness change only
        return (s.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

    def extract(self, img):          # decoder reads the attacked image at its real size, like server.py
        with self.torch.no_grad():
            return (self.dec(self._t(img))[0].numpy() > 0.5).astype(np.uint8)


def methods(args):
    m = {
        "lsb": (lambda c, b: lsb_embed(c, b), lambda img: lsb_extract(cs.restore_size(img), 1024)),
        "dct": (lambda c, b: cs.dct_embed(c, b, args.dct_strength), cs.dct_extract),
        "dwt": (lambda c, b: cs.dwt_embed(c, b, args.dwt_step), lambda img: cs.dwt_extract(img, args.dwt_step)),
    }
    if "cnn" in args.methods:
        # getattr: cross_matrix.py and sticker_test.py share this function but do not define these options
        cnn = CNN(args.checkpoint, getattr(args, "cnn_scale", None), getattr(args, "cnn_luma", False))
        m["cnn"] = (cnn.embed, cnn.extract)
    return {k: v for k, v in m.items() if k in args.methods}


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="folder of PNG/JPG photos")
    src.add_argument("--samples", action="store_true", help="use scikit-image's bundled photos")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--methods", nargs="+", default=["lsb", "dct", "dwt", "cnn"])
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--dct-strength", type=float, default=cs.DCT_STRENGTH)
    ap.add_argument("--dwt-step", type=float, default=cs.DWT_STEP)
    ap.add_argument("--cnn-scale", type=float, default=0.03, help="CNN residual strength at embed time (trained at 0.03)")
    ap.add_argument("--cnn-luma", action="store_true", help="experiment: keep only the brightness part of the CNN's change")
    ap.add_argument("--out", default="results/methods_results.csv")
    args = ap.parse_args()

    meths = methods(args)
    images = load_images(args)
    bits = encode_message(MESSAGE)
    rows, t0 = [], time.time()
    for name, cover in images:
        for mname, (embed, extract) in meths.items():
            stego = embed(cover, bits)
            q = (psnr(cover, stego, data_range=255), ssim(cover, stego, channel_axis=2, data_range=255))
            for algo, code in ALGOS.items():
                for size in SIZES:
                    got = extract(cv2.resize(stego, (size, size), interpolation=code))
                    dec = decode_payload(got)
                    rows.append({"image": name, "method": mname, "algorithm": algo, "size": size,
                                 "ber": float(np.mean(got != bits)), "exact": int(dec["ok"] and dec["text"] == MESSAGE),
                                 "embed_psnr": q[0], "embed_ssim": q[1]})
        print(f"  {name}: done ({time.time() - t0:.0f}s)", file=sys.stderr)

    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def summary(key_fn, keys, title):
        print(f"\n{title}")
        for m in meths:
            sel = [r for r in rows if r["method"] == m]
            cells = []
            for k in keys:
                s = [r for r in sel if key_fn(r) == k]
                cells.append(f"{100 * np.mean([r['ber'] for r in s]):6.2f}%")
            print(f"  {m:4s} " + " ".join(f"{c:>9s}" for c in cells))

    n = len(images)
    print(f"\n{n} images x {len(ALGOS)} algorithms x {len(SIZES)} sizes = {n * 20} attacks per method")
    print(f"DCT strength T={args.dct_strength:g}, DWT step S={args.dwt_step:g}, CNN residual scale={args.cnn_scale:g}"
          f"{', CNN brightness-only' if args.cnn_luma else ''}\n")
    print(f"  {'':4s} {'PSNR dB':>8s} {'SSIM':>7s} {'mean BER':>9s} {'worst BER':>10s} {'exact msg':>10s}")
    for m in meths:
        sel = [r for r in rows if r["method"] == m]
        per_img = {r["image"]: (r["embed_psnr"], r["embed_ssim"]) for r in sel}
        print(f"  {m:4s} {np.mean([v[0] for v in per_img.values()]):8.2f} {np.mean([v[1] for v in per_img.values()]):7.4f} "
              f"{100 * np.mean([r['ber'] for r in sel]):8.2f}% {100 * max(r['ber'] for r in sel):9.2f}% "
              f"{100 * np.mean([r['exact'] for r in sel]):9.1f}%")
    summary(lambda r: r["size"], SIZES, "Mean BER by output size:       " + "  ".join(f"{s} px" for s in SIZES))
    summary(lambda r: r["algorithm"], list(ALGOS), "Mean BER by algorithm:   " + "  ".join(f"{a:>9s}" for a in ALGOS))
    print(f"\nPer-attack rows: {args.out}")


if __name__ == "__main__":
    main()
