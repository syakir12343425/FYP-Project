"""Does a method fail more on colourful photos? Compares per-image BER with two image measures.

  colourfulness  Hasler & Suesstrunk (2003) metric M on the 512x512 cover (higher = more vivid colour;
                 roughly: <15 not colourful, 33 moderately, 59 highly, >82 extremely colourful)
  saturated      share of pixels with any channel <= 8 or >= 247 (room for the change is clipped)

Usage (repo root):
    python tools/colour_check.py results/div2k_methods_default.csv data/DIV2K_valid_HR
    python tools/colour_check.py results/div2k_classical_32db.csv data/DIV2K_valid_HR --method dct
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_methods import square_512  # noqa: E402


def colourfulness(img: np.ndarray) -> float:
    r, g, b = (img[..., i].astype(np.float64) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def saturated(img: np.ndarray) -> float:
    return float(((img <= 8) | (img >= 247)).any(axis=2).mean() * 100)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("images")
    ap.add_argument("--method", default="cnn")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    ber, ok = defaultdict(list), defaultdict(list)
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] == args.method:
                ber[r["image"]].append(float(r["ber"]))
                ok[r["image"]].append(int(r["exact"]))
    if not ber:
        raise SystemExit(f"no rows for method '{args.method}' in {args.csv}")

    rows = []
    for name in sorted(ber):
        img = square_512(np.array(Image.open(Path(args.images) / name).convert("RGB")))
        rows.append((name, 100 * np.mean(ber[name]), colourfulness(img), saturated(img), all(ok[name])))
    rows.sort(key=lambda r: -r[1])

    print(f"{args.method}: per-image mean BER vs colourfulness and saturated share ({len(rows)} images)")
    print(f"  {'image':10s} {'mean BER':>9s} {'colourful':>10s} {'saturated':>10s}  never failed?")
    for name, b, c, s, good in rows[: args.top]:
        print(f"  {name:10s} {b:8.2f}% {c:10.1f} {s:9.1f}%  {'yes' if good else 'NO'}")

    b = np.array([r[1] for r in rows])
    for i, label in ((2, "colourfulness"), (3, "saturated share")):
        v = np.array([r[i] for r in rows])
        bad = [r[i] for r in rows if not r[4]]
        good = [r[i] for r in rows if r[4]]
        line = f"\nBER vs {label:15s}: Pearson r = {np.corrcoef(b, v)[0, 1]:.2f}, Spearman rho = {spearman(b, v):.2f}"
        if bad and good:
            line += f"; failing images mean {np.mean(bad):.1f} vs never-failing {np.mean(good):.1f}"
        print(line)


if __name__ == "__main__":
    main()
