"""Message <-> 1024-bit payload codec, with whitening.

Why whitening: the CNN was trained (and benchmarked) on uniformly random bits. A real text message
is the opposite: regular ASCII patterns plus a long run of zero padding. Measured with
tools/payload_experiment.py (bicubic -> 384px), plain text gave 47.75% BER (60% in the padding)
while random bits gave 1.76%. XOR-ing the message with a fixed pseudo-random keystream makes every
payload look random to the network (1.17% BER in the same test); XOR-ing again after decoding
restores the text. Bit errors pass straight through XOR, so BER is unchanged by the unwhitening.

This is NOT encryption: the keystream is fixed and public (it is in this repo). It only reshapes the
bit statistics. Confidentiality would need a real cipher with a secret key (future work).

Error correction (rs.py): even at ~1% BER a long message loses several characters, and one flipped bit
can turn a space into the 0x00 terminator. So the 128-byte payload is a Reed-Solomon codeword:

    data block (64 B) = [length: 1 B][message: <= 63 B UTF-8][zero padding]
    codeword  (128 B) = data block + 64 parity bytes       -> repairs up to 32 wrong bytes
    payload  (1024 b) = whiten(codeword bits)

Decoding either returns the exact message (ok=True) or reports failure (ok=False); in tests with
33-44 wrong bytes it never returned a wrong message silently. Raw BER is still measured on the
1024 transmitted bits, so it stays comparable with the eval.py benchmark.
"""
import hashlib

import numpy as np

from rs import ReedSolomonError, rs_decode, rs_encode

MESSAGE_BITS = 1024                       # must match training (configs/config.yaml -> message.bits)
CODEWORD_BYTES = MESSAGE_BITS // 8        # 128
PARITY_BYTES = 64                         # RS parity: corrects up to PARITY_BYTES // 2 = 32 wrong bytes
DATA_BYTES = CODEWORD_BYTES - PARITY_BYTES  # 64 = 1 length byte + up to 63 message bytes
MAX_MESSAGE_BYTES = DATA_BYTES - 1        # 63
_LABEL = b"fyp-stego/whitening/v1"        # changing this breaks decoding of older stego images


def _keystream(n_bits: int) -> np.ndarray:
    """Deterministic pseudo-random bits from SHA-256 in counter mode (same on every machine and
    NumPy version, unlike np.random)."""
    out, counter = b"", 0
    while len(out) * 8 < n_bits:
        out += hashlib.sha256(_LABEL + counter.to_bytes(4, "big")).digest()
        counter += 1
    return np.unpackbits(np.frombuffer(out, dtype=np.uint8))[:n_bits]


KEYSTREAM = _keystream(MESSAGE_BITS)


def text_to_bits(text: str, n_bits: int = MESSAGE_BITS) -> np.ndarray:
    """Plain (un-whitened) bits: UTF-8 bytes, truncated/zero-padded to n_bits."""
    n_bytes = n_bits // 8
    data = text.encode("utf-8")[:n_bytes].ljust(n_bytes, b"\x00")
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_text(bits: np.ndarray) -> str:
    """Plain bits back to text, stopping at the zero padding."""
    byte_arr = np.packbits(np.asarray(bits, dtype=np.uint8))
    raw = byte_arr.tobytes().split(b"\x00", 1)[0]
    return raw.decode("utf-8", errors="replace")


def whiten(bits: np.ndarray) -> np.ndarray:
    """XOR with the keystream. Applying it twice gives the original bits back."""
    return (np.asarray(bits, dtype=np.uint8) ^ KEYSTREAM[: len(bits)]).astype(np.uint8)


class MessageTooLong(ValueError):
    pass


def encode_message(text: str) -> np.ndarray:
    """Text -> the 1024 whitened bits that actually get embedded."""
    data = text.encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise MessageTooLong(f"Message is {len(data)} bytes; the limit is {MAX_MESSAGE_BYTES} bytes (UTF-8).")
    block = bytes([len(data)]) + data + bytes(DATA_BYTES - 1 - len(data))
    codeword = bytes(rs_encode(block, PARITY_BYTES))
    return whiten(np.unpackbits(np.frombuffer(codeword, dtype=np.uint8)))


def decode_payload(raw_bits: np.ndarray) -> dict:
    """Decoder output bits -> {text, ok, corrected_bytes}.
    ok=False means more than 32 bytes were damaged: the text is then a best-effort, UNCORRECTED read."""
    codeword = np.packbits(whiten(raw_bits)).tolist()
    try:
        block, corrected = rs_decode(codeword, PARITY_BYTES)
        length = block[0]
        if length > MAX_MESSAGE_BYTES:
            raise ReedSolomonError("invalid length field")
        text = bytes(block[1:1 + length]).decode("utf-8", errors="replace")
        return {"text": text, "ok": True, "corrected_bytes": corrected}
    except ReedSolomonError:
        block = codeword[:DATA_BYTES]
        length = min(block[0], MAX_MESSAGE_BYTES)
        text = bytes(block[1:1 + length]).decode("utf-8", errors="replace")
        return {"text": text, "ok": False, "corrected_bytes": None}
