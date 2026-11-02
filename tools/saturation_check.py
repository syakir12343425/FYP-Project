"""Test one explanation for image-dependent failures: pixels near 0 or 255 clip the embedded change.

Usage (repo root):
    python tools/saturation_check.py results/div2k_methods_default.csv data/DIV2K_valid_HR
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

LOW, HIGH = 8, 247


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("images")
    ap.add_argument("--method", default="cnn")
    args = ap.parse_args()

    ber, ok = defaultdict(list), defaultdict(list)
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] == args.method:
                ber[r["image"]].append(float(r["ber"]))
                ok[r["image"]].append(int(r["exact"]))

    rows = []
    for name in sorted(ber):
        img = square_512(np.array(Image.open(Path(args.images) / name).convert("RGB")))
        near = ((img <= LOW) | (img >= HIGH)).any(axis=2).mean()
        rows.append((name, 100 * np.mean(ber[name]), 100 * near, all(ok[name])))
    rows.sort(key=lambda r: -r[1])

    print(f"{args.method}: mean BER vs share of near-black/near-white pixels (any channel <= {LOW} or >= {HIGH})")
    print(f"  {'image':10s} {'mean BER':>9s} {'saturated':>10s}  never failed?")
    for name, b, s, good in rows[:15]:
        print(f"  {name:10s} {b:8.2f}% {s:9.1f}%  {'yes' if good else 'NO'}")
    b, s = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    bad = [r[2] for r in rows if not r[3]]
    good = [r[2] for r in rows if r[3]]
    print(f"\nCorrelation (BER vs saturated share) over {len(rows)} images: r = {np.corrcoef(b, s)[0, 1]:.2f}")
    if bad and good:
        print(f"Saturated share: images that failed {np.mean(bad):.1f}% vs never failed {np.mean(good):.1f}%")


if __name__ == "__main__":
    main()
