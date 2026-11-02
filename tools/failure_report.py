"""Which images make a method fail? Reads a CSV written by benchmark_methods.py.

Usage (repo root):
    python tools/failure_report.py results/div2k_methods_default.csv
    python tools/failure_report.py results/div2k_methods_default.csv --method dct --top 20
"""
import argparse
import csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--method", default="cnn")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    per_image = defaultdict(list)
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] == args.method:
                per_image[r["image"]].append(r)
    if not per_image:
        raise SystemExit(f"no rows for method '{args.method}' in {args.csv}")

    stats = []
    for img, rows in per_image.items():
        bers = [float(r["ber"]) for r in rows]
        stats.append((img, 100 * sum(bers) / len(bers), 100 * max(bers),
                      sum(int(r["exact"]) for r in rows), len(rows), float(rows[0]["embed_psnr"])))
    stats.sort(key=lambda s: -s[1])

    print(f"{args.method}: {len(stats)} images, worst {args.top} by mean BER")
    print(f"  {'image':12s} {'mean BER':>9s} {'worst BER':>10s} {'exact':>8s} {'PSNR dB':>8s}")
    for img, mean, worst, ok, n, p in stats[: args.top]:
        print(f"  {img:12s} {mean:8.2f}% {worst:9.2f}% {ok:4d}/{n:<3d} {p:8.2f}")
    failing = [s for s in stats if s[3] < s[4]]
    lost = sum(s[4] - s[3] for s in stats)
    total = sum(s[4] for s in stats)
    print(f"\n{len(failing)} of {len(stats)} images lost the message at least once "
          f"({lost} of {total} attacks failed)")
    clean = [s for s in stats if s[3] == s[4]]
    if failing and clean:
        print(f"Never-failing images: mean BER {sum(s[1] for s in clean) / len(clean):.2f}%; "
              f"failing images: mean BER {sum(s[1] for s in failing) / len(failing):.2f}%")


if __name__ == "__main__":
    main()
