"""Every encoder x every decoder x every resize algorithm x every output size.

Matching pairs (encoder == decoder) are the real comparison. Mismatched pairs are a CONTROL: a decoder
looking for the wrong pattern should score about 50% BER (guessing) and must never report an exact
message, which checks that Reed-Solomon does not produce false positives.

Usage (repo root):
    python tools/cross_matrix.py --samples --methods lsb dct dwt          # no model needed
    python tools/cross_matrix.py --images data/DIV2K_valid_HR --limit 20  # all methods incl. CNN
Writes results/cross_matrix.csv and prints three tables.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import classical_stego as cs                                   # noqa: E402
from benchmark_methods import ALGOS, MESSAGE, SIZES, load_images, methods  # noqa: E402
from payload import decode_payload, encode_message             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images")
    src.add_argument("--samples", action="store_true")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--methods", nargs="+", default=["lsb", "dct", "dwt", "cnn"])
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--out", default="results/cross_matrix.csv")
    args = ap.parse_args()
    args.dct_strength, args.dwt_step = cs.DCT_STRENGTH, cs.DWT_STEP

    meths = methods(args)
    names = list(meths)
    images = load_images(args)
    bits = encode_message(MESSAGE)
    rows = []
    for img_name, cover in images:
        for enc in names:
            stego = meths[enc][0](cover, bits)
            for algo, code in ALGOS.items():
                for size in SIZES:
                    attacked = cv2.resize(stego, (size, size), interpolation=code)
                    for dec in names:
                        got = meths[dec][1](attacked)
                        d = decode_payload(got)
                        rows.append({"image": img_name, "encoder": enc, "decoder": dec, "algorithm": algo,
                                     "size": size, "ber": float(np.mean(got != bits)),
                                     "exact": int(d["ok"] and d["text"] == MESSAGE),
                                     "false_ok": int(d["ok"] and d["text"] != MESSAGE)})
        print(f"  {img_name}: done", file=sys.stderr)

    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def mean(sel, key):
        return 100 * np.mean([r[key] for r in sel]) if sel else float("nan")

    n = len(images)
    print(f"\n{n} images x {len(ALGOS)} algorithms x {len(SIZES)} sizes = {n * 20} attacks per pair\n")

    print("1) Mean BER, encoder (rows) x decoder (columns). Diagonal = real result; off-diagonal = control (~50%).")
    print("   " + "".join(f"{d:>10s}" for d in names))
    for e in names:
        print(f"   {e:4s}" + "".join(f"{mean([r for r in rows if r['encoder'] == e and r['decoder'] == d], 'ber'):9.2f}%" for d in names))
    false_ok = sum(r["false_ok"] for r in rows)
    print(f"   Wrong message reported as exact (any pair): {false_ok}\n")

    match = [r for r in rows if r["encoder"] == r["decoder"]]
    print("2) Matching pairs: mean BER  /  exact-message rate, by output size")
    print("   " + "".join(f"{f'{s} px':>18s}" for s in SIZES))
    for m in names:
        cells = []
        for s in SIZES:
            sel = [r for r in match if r["encoder"] == m and r["size"] == s]
            cells.append(f"{mean(sel, 'ber'):6.2f}% / {mean(sel, 'exact'):5.1f}%")
        print(f"   {m:4s}" + "".join(f"{c:>18s}" for c in cells))

    print("\n3) Matching pairs: mean BER by algorithm x size")
    for m in names:
        print(f"   {m.upper()}" + "".join(f"{f'{s} px':>10s}" for s in SIZES))
        for algo in ALGOS:
            print(f"     {algo:9s}" + "".join(
                f"{mean([r for r in match if r['encoder'] == m and r['algorithm'] == algo and r['size'] == s], 'ber'):9.2f}%"
                for s in SIZES))
    print(f"\nAll rows: {args.out}")


if __name__ == "__main__":
    main()
