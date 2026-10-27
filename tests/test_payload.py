"""Unit tests for payload.py (no model, no server needed).   python tests/test_payload.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from payload import (KEYSTREAM, MESSAGE_BITS, bits_to_text, decode_bits,  # noqa: E402
                     encode_message, text_to_bits)

MSGS = ["try something lagi la kawan", "harini kita nak try somthing bro", "", "x" * 200, "Selamat pagi! 😀"]


def test_roundtrip_without_errors():
    for m in MSGS:
        expected = m.encode("utf-8")[:MESSAGE_BITS // 8].decode("utf-8", errors="ignore")
        assert bits_to_text(decode_bits(encode_message(m))).startswith(expected[:120]), m


def test_whitened_payload_looks_random():
    # Plain text is ~85% zeros; the whitened payload should be close to 50/50, like training data.
    m = "try something lagi la kawan"
    plain, white = text_to_bits(m), encode_message(m)
    assert plain.mean() < 0.2
    assert 0.4 < white.mean() < 0.6, white.mean()
    # and no long constant runs: longest run of identical bits stays small
    runs = np.diff(np.flatnonzero(np.diff(np.r_[-1, white.astype(int), -1]) != 0))
    assert runs.max() < 20, runs.max()


def test_bit_errors_pass_through_unchanged():
    m = "try something lagi la kawan"
    sent = encode_message(m)
    rng = np.random.default_rng(1)
    flip = rng.random(MESSAGE_BITS) < 0.012          # ~1.2% channel errors
    received = sent ^ flip.astype(np.uint8)
    ber = np.mean(decode_bits(received) != text_to_bits(m))
    assert abs(ber - flip.mean()) < 1e-9, (ber, flip.mean())


def test_keystream_is_fixed():
    # Stego images made today must decode next year: the keystream must never silently change.
    assert KEYSTREAM.shape == (MESSAGE_BITS,)
    assert "".join(map(str, KEYSTREAM[:32])) == FIRST_32, "keystream changed - old stego images won't decode"


FIRST_32 = "10110000010000010000110101011011"  # frozen: first 32 keystream bits

if __name__ == "__main__":
    tests = [test_roundtrip_without_errors, test_whitened_payload_looks_random,
             test_bit_errors_pass_through_unchanged, test_keystream_is_fixed]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print("\nRESULT: all payload tests passed")
