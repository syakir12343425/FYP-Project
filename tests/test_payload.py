"""Unit tests for the payload codec: Reed-Solomon (rs.py) + whitening (payload.py).
No model or server needed.   python tests/test_payload.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from payload import (KEYSTREAM, MAX_MESSAGE_BYTES, MESSAGE_BITS, MessageTooLong,  # noqa: E402
                     decode_payload, encode_message)
from rs import rs_encode  # noqa: E402

MSGS = ["try something lagi la kawan", "Hidden message that survives downscaling.", "a",
        "x" * MAX_MESSAGE_BYTES, "Selamat pagi! 😀", "line one\nline two"]
FIRST_32 = "10110000010000010000110101011011"  # frozen: first 32 keystream bits


def flip(bits, rng, rate=None, positions=None):
    out = bits.copy()
    if positions is None:
        positions = np.flatnonzero(rng.random(MESSAGE_BITS) < rate)
    out[positions] ^= 1
    return out, len(positions)


def test_roundtrip_without_errors():
    for m in MSGS:
        r = decode_payload(encode_message(m))
        assert r == {"text": m, "ok": True, "corrected_bytes": 0}, (m, r)


def test_rs_matches_reference_vector():
    # Published reedsolo example: RSCodec(10).encode(b'hello world') -> independent implementation check.
    assert bytes(rs_encode(b"hello world", 10)) == b"hello world\xed%T\xc4\xfd\xfd\x89\xf3\xa8\xaa"


def test_length_limit():
    encode_message("é" * (MAX_MESSAGE_BYTES // 2))         # 2-byte characters, still within 63 bytes
    for too_long in ["x" * (MAX_MESSAGE_BYTES + 1), "😀" * 16]:   # 64 bytes each
        try:
            encode_message(too_long)
        except MessageTooLong:
            continue
        raise AssertionError(f"accepted {len(too_long.encode())} bytes")


def test_whitened_payload_looks_random():
    white = encode_message("try something lagi la kawan")
    assert 0.4 < white.mean() < 0.6, white.mean()
    runs = np.diff(np.flatnonzero(np.diff(np.r_[-1, white.astype(int), -1]) != 0))
    assert runs.max() < 20, runs.max()


def test_corrects_realistic_bit_error_rates():
    # The measured CNN BER is ~0.9-1.7% per image; 5% is the project's failure threshold.
    rng = np.random.default_rng(1)
    m = "Hidden message that survives downscaling."
    sent = encode_message(m)
    print()
    # 1-2%: must always work (covers every per-image BER measured so far, 0.88-1.66%).
    # 3%+: about 28 of 128 bytes get hit on average, close to the 32-byte limit, so recovery
    # becomes probabilistic. These rows document the edge; they are not "failures" of the code.
    for rate, need in [(0.01, 1.0), (0.02, 1.0), (0.03, 0.75), (0.04, 0.0), (0.05, 0.0)]:
        ok = 0
        for _ in range(300):
            rx, nflips = flip(sent, rng, rate)
            assert abs(np.mean(rx != sent) - nflips / MESSAGE_BITS) < 1e-12   # raw BER is what we flipped
            r = decode_payload(rx)
            assert not (r["ok"] and r["text"] != m), "returned a WRONG message marked as ok"
            ok += r["ok"]
        print(f"    random bit errors at {rate:.0%}: exact recovery {ok}/300")
        assert ok / 300 >= need


def test_clustered_errors():
    # A bad image region damages neighbouring bits; neighbouring bits share bytes, which RS likes.
    rng = np.random.default_rng(2)
    m = "line one\nline two"
    sent = encode_message(m)
    for _ in range(200):
        start = rng.integers(0, MESSAGE_BITS - 200)
        rx, _ = flip(sent, rng, positions=np.arange(start, start + 200))   # 200 consecutive bits
        bytes_hit = (start + 199) // 8 - start // 8 + 1                     # 25, or 26 if not byte-aligned
        assert decode_payload(rx) == {"text": m, "ok": True, "corrected_bytes": bytes_hit}


def test_too_many_errors_is_reported_not_hidden():
    rng = np.random.default_rng(3)
    m = "Hidden message that survives downscaling."
    sent = encode_message(m)
    failed = 0
    for _ in range(300):
        rx, _ = flip(sent, rng, 0.12)          # ~123 wrong bits -> far beyond 32 bytes
        r = decode_payload(rx)
        assert not (r["ok"] and r["text"] != m), "returned a WRONG message marked as ok"
        failed += not r["ok"]
    assert failed == 300, failed


def test_keystream_is_fixed():
    # Stego images made today must decode next year: the keystream must never silently change.
    assert KEYSTREAM.shape == (MESSAGE_BITS,)
    assert "".join(map(str, KEYSTREAM[:32])) == FIRST_32, "keystream changed - old stego images won't decode"


if __name__ == "__main__":
    tests = [test_roundtrip_without_errors, test_rs_matches_reference_vector, test_length_limit,
             test_whitened_payload_looks_random, test_corrects_realistic_bit_error_rates,
             test_clustered_errors, test_too_many_errors_is_reported_not_hidden, test_keystream_is_fixed]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print("\nRESULT: all payload tests passed")
