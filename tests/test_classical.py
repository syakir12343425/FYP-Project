"""Tests for classical_stego.py (DCT and DWT). NumPy + OpenCV + scikit-image only, no model needed.
    python tests/test_classical.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import skimage.data
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import classical_stego as cs                     # noqa: E402
from payload import decode_payload, encode_message  # noqa: E402

MSG = "Meeting at 3pm, room B2"
COVER = skimage.data.astronaut()                 # a real 512x512 RGB photo, bundled with scikit-image
METHODS = {"dct": (cs.dct_embed, cs.dct_extract), "dwt": (cs.dwt_embed, cs.dwt_extract)}


def test_transforms_are_exact():
    rng = np.random.default_rng(0)
    y = rng.random((512, 512)) * 255
    ll, details = cs._ll4(y)
    rec = ll
    for d in reversed(details):
        rec = cs.haar_idwt2(rec, d)
    assert np.abs(rec - y).max() < 1e-9
    assert np.allclose(ll, 16 * y.reshape(32, 16, 32, 16).mean((1, 3)))     # LL4 = 16 x block mean
    assert np.allclose(cs._D8 @ cs._D8.T, np.eye(8))                          # orthonormal DCT


def test_roundtrip_without_attack():
    bits = encode_message(MSG)
    for name, (embed, extract) in METHODS.items():
        stego = embed(COVER, bits)
        assert stego.dtype == np.uint8 and stego.shape == COVER.shape
        # Not always 0 raw errors: in saturated blocks (pixels already at 0 or 255) part of the change is
        # clipped away. On this photo that costs DCT 2 of 1024 bits; Reed-Solomon repairs them.
        assert np.mean(extract(stego) != bits) <= 0.01, name
        d = decode_payload(extract(stego))
        assert d["ok"] and d["text"] == MSG, (name, d)


def test_image_quality_near_target():
    bits = encode_message(MSG)
    for name, (embed, _) in METHODS.items():
        q = psnr(COVER, embed(COVER, bits), data_range=255)
        assert 38 < q < 45, (name, q)          # defaults chosen for ~40.7 dB


def test_survives_averaging_resizes():
    bits = encode_message(MSG)
    for name, (embed, extract) in METHODS.items():
        stego = embed(COVER, bits)
        for algo in (cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_AREA):
            for size in (384, 299, 256, 224):
                d = decode_payload(extract(cv2.resize(stego, (size, size), interpolation=algo)))
                assert d["ok"] and d["text"] == MSG, (name, algo, size, d)


def test_dwt_nearest_weakness_is_reported_not_hidden():
    # Known limitation: nearest-neighbour does not average, so the DWT block means are lost.
    # The decoder must then REPORT failure, never return a wrong message marked ok.
    stego = cs.dwt_embed(COVER, encode_message(MSG))
    d = decode_payload(cs.dwt_extract(cv2.resize(stego, (224, 224), interpolation=cv2.INTER_NEAREST)))
    assert not (d["ok"] and d["text"] != MSG)


if __name__ == "__main__":
    for t in [test_transforms_are_exact, test_roundtrip_without_attack, test_image_quality_near_target,
              test_survives_averaging_resizes, test_dwt_nearest_weakness_is_reported_not_hidden]:
        t()
        print(f"[PASS] {t.__name__}")
    print("\nRESULT: all classical method tests passed")
