"""Can a hidden message survive being sent as a WhatsApp sticker? (supervisor question)

A WhatsApp static sticker must be a 512x512 WebP file of at most 100 KB. That size limit forces LOSSY WebP
compression, which no method here was designed or trained for. For every method this tool:
  1. embeds the same payload the server uses (Reed-Solomon + whitening),
  2. saves the stego image as WebP at the HIGHEST quality that fits in 100 KB (what a careful sender would do),
  3. reloads it and extracts, reporting raw BER and whether the message came back exact.
It also reports, for reference, lossless WebP (a sticker app that kept every pixel, if it fit) and plain JPEG at
quality 75 (a rough stand-in for WhatsApp "photo" mode, whose exact settings are not published).

Usage (repo root):
    python tools/sticker_test.py --samples --methods lsb dct dwt       # no model needed
    python tools/sticker_test.py --images "C:\\path\\to\\photos" --limit 5   # all methods incl. CNN
Results are printed only; the WebP files are written to results/stickers/ so you can try sending one.
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_methods import MESSAGE, load_images, methods  # noqa: E402
from payload import decode_payload, encode_message           # noqa: E402

LIMIT = 100 * 1024


def webp_under_limit(img: np.ndarray):
    """Highest WebP quality (100..5) whose file is <= 100 KB. Returns (bytes, quality) or (None, None)."""
    for q in range(100, 4, -5):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "WEBP", quality=q, method=6)
        if buf.tell() <= LIMIT:
            return buf.getvalue(), q
    return None, None


def roundtrip(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def encode(img, fmt, **kw):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, fmt, **kw)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images")
    src.add_argument("--samples", action="store_true")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--methods", nargs="+", default=["lsb", "dct", "dwt", "cnn"])
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--dct-strength", type=float, default=None)
    ap.add_argument("--dwt-step", type=float, default=None)
    ap.add_argument("--out-dir", default="results/stickers")
    args = ap.parse_args()
    import classical_stego as cs
    args.dct_strength = args.dct_strength or cs.DCT_STRENGTH
    args.dwt_step = args.dwt_step or cs.DWT_STEP

    meths = methods(args)
    images = load_images(args)
    bits = encode_message(MESSAGE)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {m: {k: [] for k in ("sticker", "lossless", "jpeg75")} for m in meths}
    quals, sizes = [], []

    for name, cover in images:
        for m, (embed, extract) in meths.items():
            stego = embed(cover, bits)
            data, q = webp_under_limit(stego)
            quals.append(q)
            sizes.append(len(data) if data else None)
            if data:
                (out_dir / f"{Path(name).stem}_{m}.webp").write_bytes(data)
            cases = {"sticker": roundtrip(data) if data else None,
                     "lossless": roundtrip(encode(stego, "WEBP", lossless=True)),
                     "jpeg75": roundtrip(encode(stego, "JPEG", quality=75))}
            for case, img in cases.items():
                if img is None:
                    continue
                got = extract(img)
                d = decode_payload(got)
                stats[m][case].append((float(np.mean(got != bits)), d["ok"] and d["text"] == MESSAGE))

    n = len(images)
    q_ok = [q for q in quals if q]
    print(f"\n{n} images. Sticker = 512x512 WebP <= 100 KB at the highest quality that fits "
          f"(quality used: {min(q_ok)}-{max(q_ok)}, median {int(np.median(q_ok))}).\n")
    print(f"  {'method':6s} {'sticker BER':>12s} {'exact':>7s}   {'lossless BER':>12s} {'exact':>7s}   {'JPEG75 BER':>11s} {'exact':>7s}")
    for m in meths:
        cells = []
        for case in ("sticker", "lossless", "jpeg75"):
            r = stats[m][case]
            cells.append(f"{100 * np.mean([b for b, _ in r]):11.2f}% {sum(e for _, e in r):3d}/{len(r):<3d}")
        print(f"  {m:6s} " + "   ".join(cells))
    print(f"\nWebP stickers saved in {out_dir}/ (open one in WhatsApp to try it for real).")


if __name__ == "__main__":
    main()
