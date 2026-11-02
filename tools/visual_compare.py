"""Side-by-side visual comparison of the embedding methods on one photo (thesis figure).

Saves results/visual/<image>_compare.png: top row = cover and each stego image (a zoomed crop),
bottom row = |stego - cover| amplified x10, so the *pattern* of each method's change is visible.
PSNR/SSIM printed on each panel are measured on the full 512x512 image.

Usage (repo root):
    python tools/visual_compare.py --image data/DIV2K_valid_HR/0801.png
    python tools/visual_compare.py --image data/DIV2K_valid_HR/0801.png --dct-strength 140 --dwt-step 384
    python tools/visual_compare.py --sample astronaut --methods dct dwt          # no CNN / no torch needed
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import classical_stego as cs                       # noqa: E402
from benchmark_methods import CNN, MESSAGE, square_512  # noqa: E402
from lsb_baseline import lsb_embed                 # noqa: E402
from payload import encode_message                 # noqa: E402

CROP, ZOOM, AMP = 128, 3, 10     # 128x128 crop shown at 3x; differences amplified 10x


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 30], fill=(0, 0, 0))
    d.text((6, 8), text, fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to a photo")
    src.add_argument("--sample", help="scikit-image sample name, e.g. astronaut")
    ap.add_argument("--methods", nargs="+", default=["lsb", "dct", "dwt", "cnn"])
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--dct-strength", type=float, default=cs.DCT_STRENGTH)
    ap.add_argument("--dwt-step", type=float, default=cs.DWT_STEP)
    ap.add_argument("--cnn-scale", type=float, default=0.03)
    ap.add_argument("--cnn-luma", action="store_true", help="keep only the brightness part of the CNN's change")
    ap.add_argument("--crop", type=int, nargs=2, metavar=("X", "Y"), default=None,
                    help="top-left of the zoom crop in the 512 image (default: centre)")
    ap.add_argument("--out-dir", default="results/visual")
    args = ap.parse_args()

    if args.image:
        name = Path(args.image).stem
        cover = square_512(np.array(Image.open(args.image).convert("RGB")))
    else:
        import skimage.data as d
        name = args.sample
        cover = square_512(getattr(d, args.sample)())

    bits = encode_message(MESSAGE)
    embed = {
        "lsb": lambda c: lsb_embed(c, bits),
        "dct": lambda c: cs.dct_embed(c, bits, args.dct_strength),
        "dwt": lambda c: cs.dwt_embed(c, bits, args.dwt_step),
    }
    if "cnn" in args.methods:
        cnn = CNN(args.checkpoint, args.cnn_scale, args.cnn_luma)
        embed["cnn"] = lambda c: cnn.embed(c, bits)
    setting = {"lsb": "", "dct": f" T={args.dct_strength:g}", "dwt": f" S={args.dwt_step:g}",
               "cnn": f" scale={args.cnn_scale:g}" + (" luma" if args.cnn_luma else "")}

    x, y = args.crop if args.crop else ((512 - CROP) // 2, (512 - CROP) // 2)
    crop = lambda a: a[y:y + CROP, x:x + CROP]                     # noqa: E731
    big = lambda a: Image.fromarray(a).resize((CROP * ZOOM, CROP * ZOOM), Image.NEAREST)  # noqa: E731

    tops, bottoms = [label(big(crop(cover)), "cover")], [label(big(np.zeros_like(crop(cover))), "difference x10")]
    for m in args.methods:
        stego = embed[m](cover)
        p, s = psnr(cover, stego, data_range=255), ssim(cover, stego, channel_axis=2, data_range=255)
        diff = np.clip(np.abs(stego.astype(int) - cover.astype(int)) * AMP, 0, 255).astype(np.uint8)
        tops.append(label(big(crop(stego)), f"{m.upper()}{setting[m]}  {p:.2f} dB  SSIM {s:.4f}"))
        bottoms.append(label(big(crop(diff)), f"{m.upper()} |change| x{AMP}"))
        print(f"  {m:4s}{setting[m]:12s} PSNR {p:6.2f} dB  SSIM {s:.4f}")

    w, h = CROP * ZOOM, CROP * ZOOM
    sheet = Image.new("RGB", (w * len(tops), h * 2), "white")
    for i, (t, b) in enumerate(zip(tops, bottoms)):
        sheet.paste(t, (i * w, 0))
        sheet.paste(b, (i * w, h))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.out_dir) / f"{name}_compare.png"
    sheet.save(out)
    print(f"Saved {out}  (crop at x={x}, y={y}, {CROP}x{CROP} shown at {ZOOM}x)")


if __name__ == "__main__":
    main()
