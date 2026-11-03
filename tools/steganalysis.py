"""Steganalysis (FYP item 4): can someone who does NOT know the message tell that an image carries one?

Three kinds of attacker, from weakest to strongest:

  1. Visual (what Aperi'Solve shows): bit planes and an amplified difference image.
     -> figure  results/steganalysis/visual_<image>.png
  2. Targeted statistical attacks on LSB replacement:
       chi-square attack (Westfeld & Pfitzmann 2000), on the whole image and on the first 1,024 samples
       in scan order (exactly where this project's LSB writes), and RS analysis (Fridrich et al. 2001),
       which estimates the fraction of LSBs that were changed.
  2b. An attacker who knows the algorithm (Kerckhoffs's principle) but not the message: run the public
     decoder and check whether Reed-Solomon accepts the result. There is no secret key in this system,
     so this is the strongest attack of all; it also measures false alarms on clean covers.
  3. A trained blind detector: second-order SPAM features (Pevny, Bas & Fridrich 2010, T=3, 686 per channel)
     + logistic regression, cross-validated with covers and their stegos always in the same fold.
     Two feature sets: luminance only (classic, greyscale), and Y+Cb+Cr (colour-aware, 2058 features),
     because the CNN hides mostly in colour (Results log 6b-6d).
     Score: AUC and P_E = min over thresholds of (false alarms + misses) / 2.  P_E 0.5 = coin flip
     (undetectable), P_E 0 = always detected.

Every stego carries a DIFFERENT random message (random length, 10..63 bytes) through the same payload.py
as the app, so the detector cannot learn one fixed bit pattern. Payload = 1024 bits per 512x512 image
= 0.0039 bits per pixel, which is very small; keep that in mind when comparing with papers that test 0.1-0.4 bpp.

Usage (repo root; needs scikit-learn: pip install scikit-learn):
    python tools/steganalysis.py --images data/DIV2K_valid_HR --limit 100 --checkpoint checkpoints_channel_l300/best.pt
    python tools/steganalysis.py --samples --methods lsb dct dwt          # quick run without torch
Writes results/steganalysis/{summary.csv, per_image.csv, summary.json, visual_*.png}.
"""
import argparse
import csv
import json
import string
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import classical_stego as cs                                    # noqa: E402
from benchmark_methods import load_images, methods as make_methods   # noqa: E402
from payload import MAX_MESSAGE_BYTES, decode_payload, encode_message   # noqa: E402

T = 3   # SPAM truncation threshold


# ------------------------------------------------------------------ features
def _spam_one_direction(x: np.ndarray) -> np.ndarray:
    """Second-order Markov transition probabilities of truncated horizontal differences (left -> right)."""
    d = x[:, :-1] - x[:, 1:]
    d = np.clip(d, -T, T).astype(np.int64) + T
    a, b, c = d[:, :-2].ravel(), d[:, 1:-1].ravel(), d[:, 2:].ravel()
    k = 2 * T + 1
    joint = np.bincount((a * k + b) * k + c, minlength=k ** 3).astype(np.float64).reshape(k, k, k)
    cond = joint.sum(axis=2, keepdims=True)
    return np.divide(joint, cond, out=np.zeros_like(joint), where=cond > 0).ravel()


def _spam_diag(x: np.ndarray) -> np.ndarray:
    """Same along the main diagonal (top-left -> bottom-right)."""
    d = x[:-1, :-1] - x[1:, 1:]
    d = np.clip(d, -T, T).astype(np.int64) + T
    a, b, c = d[:-2, :-2].ravel(), d[1:-1, 1:-1].ravel(), d[2:, 2:].ravel()
    k = 2 * T + 1
    joint = np.bincount((a * k + b) * k + c, minlength=k ** 3).astype(np.float64).reshape(k, k, k)
    cond = joint.sum(axis=2, keepdims=True)
    return np.divide(joint, cond, out=np.zeros_like(joint), where=cond > 0).ravel()


def spam686(channel: np.ndarray) -> np.ndarray:
    """SPAM (second order, T=3): average of the 4 straight directions and of the 4 diagonals -> 2 x 343."""
    x = channel.astype(np.float64)
    straight = [x, x[:, ::-1], x.T, x.T[:, ::-1]]                    # right, left, down, up
    diag = [x, x[:, ::-1], x[::-1, :], x[::-1, ::-1]]                 # the four diagonal directions
    f1 = np.mean([_spam_one_direction(v) for v in straight], axis=0)
    f2 = np.mean([_spam_diag(v) for v in diag], axis=0)
    return np.concatenate([f1, f2])


def ycbcr(rgb: np.ndarray):
    """BT.601 full-range Y, Cb, Cr, rounded to integers like an 8-bit conversion."""
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.round(y), np.round(cb), np.round(cr)


def features(rgb: np.ndarray) -> dict:
    y, cb, cr = ycbcr(rgb)
    fy = spam686(y)
    return {"luma": fy, "colour": np.concatenate([fy, spam686(cb), spam686(cr)])}


# ------------------------------------------------------------------ targeted LSB attacks
def chi_square_p(values: np.ndarray) -> float:
    """Westfeld-Pfitzmann chi-square attack. Returns the probability of embedding (near 1 = LSBs look
    randomised by embedding, near 0 = natural)."""
    from scipy.stats import chi2
    h = np.bincount(values.ravel().astype(np.int64), minlength=256).astype(np.float64)
    even, odd = h[0::2], h[1::2]
    expected = (even + odd) / 2
    keep = expected > 4
    if keep.sum() < 2:
        return float("nan")
    stat = np.sum((even[keep] - expected[keep]) ** 2 / expected[keep])
    return float(1 - chi2.cdf(stat, keep.sum() - 1))


def _flip(x, kind):
    if kind == 1:    # F1: 2n <-> 2n+1
        return x ^ 1
    return ((x + 1) ^ 1) - 1   # F-1: 2n <-> 2n-1


def _rs_counts(ch: np.ndarray, mask=(0, 1, 1, 0)):
    g = ch[:, : ch.shape[1] // 4 * 4].astype(np.int64).reshape(-1, 4)
    m = np.array(mask, dtype=bool)

    def smooth(v):
        return np.abs(np.diff(v, axis=1)).sum(axis=1)

    def apply(kind, neg=False):
        v = g.copy()
        v[:, m] = _flip(v[:, m], -1 if neg else 1)
        return v

    f0 = smooth(g)
    out = {}
    for name, neg in (("p", False), ("n", True)):
        f = smooth(apply(1, neg))
        out["R" + name], out["S" + name] = np.mean(f > f0), np.mean(f < f0)
    return out


def rs_estimate(rgb: np.ndarray) -> float:
    """RS analysis: estimated fraction of samples with a changed LSB, averaged over R, G, B (0 = clean)."""
    ests = []
    for i in range(3):
        ch = rgb[..., i].astype(np.int64)
        a = _rs_counts(ch)
        b = _rs_counts(ch ^ 1)          # same counts after flipping every LSB
        d0, d1 = a["Rp"] - a["Sp"], b["Rp"] - b["Sp"]
        n0, n1 = a["Rn"] - a["Sn"], b["Rn"] - b["Sn"]
        A, B, C = 2 * (d1 + d0), (n0 - n1 - d1 - 3 * d0), (d0 - n0)
        if abs(A) < 1e-12:
            continue
        disc = B * B - 4 * A * C
        if disc < 0:
            continue
        roots = [(-B + s * np.sqrt(disc)) / (2 * A) for s in (1, -1)]
        z = min(roots, key=abs)
        ests.append(z / (z - 0.5))
    return float(np.clip(np.mean(ests), 0, 1)) if ests else float("nan")


# ------------------------------------------------------------------ detector
def evaluate_detector(Xc, Xs, folds=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n = len(Xc)
    X = np.vstack([Xc, Xs])
    y = np.r_[np.zeros(n), np.ones(n)]
    groups = np.r_[np.arange(n), np.arange(n)]          # a cover and its stego never split across folds
    folds = min(folds, n)
    aucs, pes, accs = [], [], []
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        inner = GroupKFold(n_splits=min(3, len(np.unique(groups[tr]))))
        model = GridSearchCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
            {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0]},
            cv=inner.split(X[tr], y[tr], groups[tr]), scoring="roc_auc")
        model.fit(X[tr], y[tr])
        s = model.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], s))
        fpr, tpr, _ = roc_curve(y[te], s)
        pes.append(float(np.min(0.5 * (fpr + 1 - tpr))))
        accs.append(float(np.mean((s > 0.5) == y[te])))
    return {"auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "pe": float(np.mean(pes)), "pe_std": float(np.std(pes)), "acc": float(np.mean(accs)), "folds": folds}


# ------------------------------------------------------------------ visual (Aperi'Solve style)
def visual_figure(name, cover, stegos, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("cover", cover)] + list(stegos.items())
    fig, ax = plt.subplots(len(rows), 4, figsize=(12, 3 * len(rows)))
    for r, (label, img) in enumerate(rows):
        diff = np.abs(img.astype(np.int16) - cover.astype(np.int16)).sum(axis=2)
        panels = [(img, f"{label}"),
                  ((img[..., 0] & 1) * 255, "red channel, bit 0"),
                  (((img[..., 0] >> 1) & 1) * 255, "red channel, bit 1"),
                  (np.clip(diff * 10, 0, 255).astype(np.uint8), "|stego - cover| x 10")]
        for c, (p, title) in enumerate(panels):
            ax[r, c].imshow(p, cmap=None if p.ndim == 3 else "gray", vmin=0, vmax=255, interpolation="nearest")
            ax[r, c].set_title(title, fontsize=9)
            ax[r, c].axis("off")
    fig.suptitle(f"Visual inspection, {name} (bit planes as shown by Aperi'Solve)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def random_message(rng: np.random.Generator) -> str:
    n = int(rng.integers(10, MAX_MESSAGE_BYTES + 1))
    alphabet = string.ascii_letters + string.digits + " .,!?-"
    return "".join(alphabet[i] for i in rng.integers(0, len(alphabet), n))


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="folder of PNG/JPG photos (e.g. data/DIV2K_valid_HR)")
    src.add_argument("--samples", action="store_true", help="use scikit-image's bundled photos (smoke test only)")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--methods", nargs="+", default=["lsb", "dct", "dwt", "cnn"])
    ap.add_argument("--checkpoint", default="checkpoints_channel_l300/best.pt")
    ap.add_argument("--dct-strength", type=float, default=cs.DCT_STRENGTH)
    ap.add_argument("--dwt-step", type=float, default=cs.DWT_STEP)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--visual", type=int, default=2, help="how many images get an Aperi'Solve-style figure")
    ap.add_argument("--out-dir", default="results/steganalysis")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    images = load_images(args)
    meths = make_methods(args)
    print(f"{len(images)} covers, methods: {', '.join(meths)}; payload 1024 bits = {1024 / 512 ** 2:.4f} bpp")

    feats = {"cover": {"luma": [], "colour": []}, **{m: {"luma": [], "colour": []} for m in meths}}
    per_image = []
    t0 = time.time()
    for i, (name, cover) in enumerate(images):
        f = features(cover)
        for k in f:
            feats["cover"][k].append(f[k])
        row = {"image": name, "cover_chi_all": chi_square_p(cover), "cover_chi_first1pct": chi_square_p(cover.ravel()[:1024]),
               "cover_rs": rs_estimate(cover)}
        stegos = {}
        for m, (embed, extract) in meths.items():
            msg = random_message(rng)
            stego = embed(cover, encode_message(msg))
            ok = decode_payload(extract(stego))["text"] == msg     # sanity: the message really is in there
            stegos[m] = stego
            f = features(stego)
            for k in f:
                feats[m][k].append(f[k])
            row.update({f"{m}_chi_all": chi_square_p(stego), f"{m}_chi_first1pct": chi_square_p(stego.ravel()[:1024]),
                        f"{m}_rs": rs_estimate(stego), f"{m}_readable": ok,
                        f"{m}_false_alarm_on_cover": bool(decode_payload(extract(cover))["ok"])})
        per_image.append(row)
        if i < args.visual:
            visual_figure(name, cover, stegos, out / f"visual_{Path(name).stem}.png")
        if (i + 1) % 10 == 0 or i + 1 == len(images):
            print(f"  {i + 1}/{len(images)} images, {time.time() - t0:.0f}s")

    with open(out / "per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_image[0]))
        w.writeheader()
        w.writerows(per_image)

    summary = []
    for m in meths:
        def mean(key):
            return float(np.nanmean([r[key] for r in per_image]))
        res = {"method": m, "n_covers": len(images),
               "readable_pct": 100 * float(np.mean([r[f"{m}_readable"] for r in per_image])),
               "known_algo_false_alarm_pct": 100 * float(np.mean([r[f"{m}_false_alarm_on_cover"] for r in per_image])),
               "chi_all_cover": mean("cover_chi_all"), "chi_all_stego": mean(f"{m}_chi_all"),
               "chi_first1pct_cover": mean("cover_chi_first1pct"), "chi_first1pct_stego": mean(f"{m}_chi_first1pct"),
               "rs_cover": mean("cover_rs"), "rs_stego": mean(f"{m}_rs")}
        for k in ("luma", "colour"):
            d = evaluate_detector(np.array(feats["cover"][k]), np.array(feats[m][k]), args.folds, args.seed)
            res.update({f"spam_{k}_auc": d["auc"], f"spam_{k}_auc_std": d["auc_std"],
                        f"spam_{k}_pe": d["pe"], f"spam_{k}_pe_std": d["pe_std"], f"spam_{k}_acc": d["acc"]})
        summary.append(res)

    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    (out / "summary.json").write_text(json.dumps({"args": vars(args), "summary": summary,
                                                  "date": time.strftime("%Y-%m-%d")}, indent=1))

    print("\nTargeted LSB attacks (mean over images; cover / stego)")
    print(f"{'method':8s} {'chi2 whole':>16s} {'chi2 first 1024':>16s} {'RS est. rate':>16s}")
    for r in summary:
        print(f"{r['method']:8s} {r['chi_all_cover']:7.3f} / {r['chi_all_stego']:6.3f} "
              f"{r['chi_first1pct_cover']:7.3f} / {r['chi_first1pct_stego']:6.3f} {r['rs_cover']:7.4f} / {r['rs_stego']:6.4f}")
    print(f"\nTrained detector, SPAM + logistic regression, {summary[0]['n_covers']} cover/stego pairs, "
          f"{min(args.folds, len(images))}-fold CV  (P_E 0.50 = undetectable, 0.00 = always caught)")
    print(f"{'method':8s} {'luma: AUC':>12s} {'P_E':>12s}   {'colour: AUC':>12s} {'P_E':>12s}   readable")
    for r in summary:
        print(f"{r['method']:8s} {r['spam_luma_auc']:6.3f}±{r['spam_luma_auc_std']:.3f} {r['spam_luma_pe']:6.3f}±{r['spam_luma_pe_std']:.3f}   "
              f"{r['spam_colour_auc']:6.3f}±{r['spam_colour_auc_std']:.3f} {r['spam_colour_pe']:6.3f}±{r['spam_colour_pe_std']:.3f}   "
              f"{r['readable_pct']:.0f}%")
    print("\nAttacker who knows the algorithm (no key exists): detects = decoder + Reed-Solomon accept")
    for r in summary:
        print(f"{r['method']:8s} detected {r['readable_pct']:.0f}% of stegos, false alarms on {r['known_algo_false_alarm_pct']:.0f}% of covers")
    print(f"\nWritten to {out}/")


if __name__ == "__main__":
    main()
