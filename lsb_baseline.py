"""
LSB baseline for comparison (proposal Experiment 1, Section 3.5.2).
Operates on uint8 numpy images, same payload size as the CNN for a fair
comparison (config.message.bits, default 1024 bits = 128 bytes).
"""
import numpy as np


def lsb_embed(cover_uint8: np.ndarray, message_bits: np.ndarray) -> np.ndarray:
    """cover_uint8: HxWx3 uint8. message_bits: 1D array of 0/1, length <= H*W*3."""
    flat = cover_uint8.flatten().copy()
    n = len(message_bits)
    assert n <= flat.size, "message too large for this cover image at 1 bit/channel"
    flat[:n] = (flat[:n] & 0xFE) | message_bits.astype(np.uint8)
    return flat.reshape(cover_uint8.shape)


def lsb_extract(stego_uint8: np.ndarray, n_bits: int) -> np.ndarray:
    flat = stego_uint8.flatten()
    return (flat[:n_bits] & 1).astype(np.uint8)
