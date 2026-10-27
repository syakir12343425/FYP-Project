"""Message <-> 1024-bit payload codec, with whitening.

Why whitening: the CNN was trained (and benchmarked) on uniformly random bits. A real text message
is the opposite: regular ASCII patterns plus a long run of zero padding. Measured with
tools/payload_experiment.py (bicubic -> 384px), plain text gave 47.75% BER (60% in the padding)
while random bits gave 1.76%. XOR-ing the message with a fixed pseudo-random keystream makes every
payload look random to the network (1.17% BER in the same test); XOR-ing again after decoding
restores the text. Bit errors pass straight through XOR, so BER is unchanged by the unwhitening.

This is NOT encryption: the keystream is fixed and public (it is in this repo). It only reshapes the
bit statistics. Confidentiality would need a real cipher with a secret key (future work).
"""
import hashlib

import numpy as np

MESSAGE_BITS = 1024                       # must match training (configs/config.yaml -> message.bits)
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


def encode_message(text: str) -> np.ndarray:
    """What actually gets embedded."""
    return whiten(text_to_bits(text))


def decode_bits(raw_bits: np.ndarray) -> np.ndarray:
    """Decoder output -> plain message bits (compare these with text_to_bits for BER)."""
    return whiten(raw_bits)
