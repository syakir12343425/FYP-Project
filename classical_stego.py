"""Classical frequency-domain steganography: DCT and DWT, for comparison with LSB and the CNN.

Both carry the same 1024-bit payload as the CNN (payload.py: Reed-Solomon codeword + whitening),
work on 512x512 RGB uint8 covers, and change only luminance. Changing luminance Y by d changes R, G and
B by exactly d each (BT.601 inverse: R = Y + 1.402(Cr-128), G = Y - ..., B = Y + 1.772(Cb-128)), so the
methods add the same per-pixel offset to all three channels.

DCT ("textbook", Koch & Zhao style)
    The 512x512 luminance is split into 8x8 blocks (64x64 blocks). Bit k of the 32x32 grid goes in
    block (2i, 2j). In that block the DCT coefficients (2,3) and (3,2) are reordered so that
    C(2,3) - C(3,2) >= +T for a 1 and <= -T for a 0. These basis functions sit at 0.125-0.19 cycles per
    pixel. The smallest tested size (224 px, 44%) keeps frequencies up to 0.22 cycles per pixel, so the
    pair survives every tested size, with more errors as the output gets smaller (measured: 0.9% BER at
    384 px, 8.6% at 224 px, T=30, 12 photos).

DWT (LL-band quantisation index modulation)
    A 4-level orthonormal Haar transform of the luminance gives a 32x32 approximation band (LL4): one
    coefficient per 16x16 pixel block, equal to 16 x the block mean. Bit k is embedded in LL4[k] with
    QIM (step S): the coefficient is moved to the nearest point of the lattice for that bit. Only LL4
    changes, so the inverse transform adds a constant offset of at most S/32 grey levels to each 16x16
    block. Block averages are what averaging resizes preserve, so it is near-perfect under bilinear, bicubic,
    Lanczos and area. Nearest-neighbour does not average (it keeps single pixels), so the block mean must be
    estimated from scattered raw pixels and errors rise sharply (measured ~19% BER). That is a real,
    reportable weakness of the method.

Clipping: where pixels are already at 0 or 255, part of the change is cut off when the stego image is
rounded to uint8, so a few bits can be wrong even with no attack (2 of 1024 for DCT on scikit-image's
astronaut photo). Reed-Solomon in payload.py repairs these.

Extraction: an attacked (smaller) image is first resized back to 512x512 with bicubic interpolation.
The receiver does not know which algorithm the attacker used, so a fixed choice is the honest setting.
"""
import cv2
import numpy as np

SIZE = 512                 # cover size, same as the CNN
GRID = 32                  # 32 x 32 = 1024 bits
DCT_STRENGTH = 50.0        # T: coefficient gap. 50 -> ~40.7 dB PSNR, about equal to DWT_STEP=128 (see the sweep)
DWT_STEP = 128.0           # S: QIM step in LL4 units (= 8 grey levels of block mean) -> ~40.8 dB PSNR
_DCT_PAIR = ((2, 3), (3, 2))
_HAAR_LEVELS = 4           # 512 / 2**4 = 32


# ---------------------------------------------------------------- shared helpers
def luminance(rgb_uint8: np.ndarray) -> np.ndarray:
    rgb = rgb_uint8.astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def add_luma_offset(rgb_uint8: np.ndarray, delta: np.ndarray) -> np.ndarray:
    out = rgb_uint8.astype(np.float64) + delta[..., None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def restore_size(img_uint8: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Attacked image back to the cover size, so block positions line up again."""
    if img_uint8.shape[0] == size and img_uint8.shape[1] == size:
        return img_uint8
    return cv2.resize(img_uint8, (size, size), interpolation=cv2.INTER_CUBIC)


def _check(cover: np.ndarray, bits: np.ndarray):
    assert cover.shape == (SIZE, SIZE, 3) and cover.dtype == np.uint8, "cover must be 512x512x3 uint8"
    assert bits.shape == (GRID * GRID,), "payload must be 1024 bits"


# ---------------------------------------------------------------- DCT
def _dct_matrix(n: int = 8) -> np.ndarray:
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m                                   # orthonormal DCT-II: C = M B M^T, B = M^T C M


_D8 = _dct_matrix(8)


def _selected_blocks(y: np.ndarray) -> np.ndarray:
    """(32, 32, 8, 8) view of the 8x8 blocks at even block rows/cols."""
    blocks = y.reshape(SIZE // 8, 8, SIZE // 8, 8).transpose(0, 2, 1, 3)
    return blocks[::2, ::2]


def dct_embed(cover: np.ndarray, bits: np.ndarray, strength: float = DCT_STRENGTH) -> np.ndarray:
    _check(cover, bits)
    y = luminance(cover)
    blocks = _selected_blocks(y)
    coef = np.einsum("ij,abjk,lk->abil", _D8, blocks, _D8)
    (u1, v1), (u2, v2) = _DCT_PAIR
    c1, c2 = coef[..., u1, v1], coef[..., u2, v2]
    sign = np.where(bits.reshape(GRID, GRID) == 1, 1.0, -1.0)
    gap = (c1 - c2) * sign                      # >= strength when the bit is already right
    mid = (c1 + c2) / 2
    need = gap < strength
    new1 = np.where(need, mid + sign * strength / 2, c1)
    new2 = np.where(need, mid - sign * strength / 2, c2)
    dcoef = np.zeros_like(coef)
    dcoef[..., u1, v1] = new1 - c1
    dcoef[..., u2, v2] = new2 - c2
    dblock = np.einsum("ji,abjk,kl->abil", _D8, dcoef, _D8)
    delta = np.zeros((SIZE // 8, SIZE // 8, 8, 8))
    delta[::2, ::2] = dblock
    delta = delta.transpose(0, 2, 1, 3).reshape(SIZE, SIZE)
    return add_luma_offset(cover, delta)


def dct_extract(img: np.ndarray) -> np.ndarray:
    y = luminance(restore_size(img))
    coef = np.einsum("ij,abjk,lk->abil", _D8, _selected_blocks(y), _D8)
    (u1, v1), (u2, v2) = _DCT_PAIR
    return (coef[..., u1, v1] > coef[..., u2, v2]).astype(np.uint8).reshape(-1)


# ---------------------------------------------------------------- DWT (Haar)
def haar_dwt2(x: np.ndarray):
    """One level of the orthonormal 2-D Haar transform -> (LL, (LH, HL, HH))."""
    a, b = x[0::2, 0::2], x[0::2, 1::2]
    c, d = x[1::2, 0::2], x[1::2, 1::2]
    return (a + b + c + d) / 2, ((a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2)


def haar_idwt2(ll: np.ndarray, details) -> np.ndarray:
    lh, hl, hh = details
    out = np.empty((ll.shape[0] * 2, ll.shape[1] * 2))
    out[0::2, 0::2] = (ll + lh + hl + hh) / 2
    out[0::2, 1::2] = (ll - lh + hl - hh) / 2
    out[1::2, 0::2] = (ll + lh - hl - hh) / 2
    out[1::2, 1::2] = (ll - lh - hl + hh) / 2
    return out


def _ll4(y: np.ndarray):
    details, ll = [], y
    for _ in range(_HAAR_LEVELS):
        ll, d = haar_dwt2(ll)
        details.append(d)
    return ll, details


def _qim(c: np.ndarray, bits: np.ndarray, step: float) -> np.ndarray:
    offset = bits * step / 2
    return step * np.round((c - offset) / step) + offset


def _qim_read(c: np.ndarray, step: float) -> np.ndarray:
    frac = np.mod(c / step, 1.0)
    return ((frac >= 0.25) & (frac < 0.75)).astype(np.uint8)


def dwt_embed(cover: np.ndarray, bits: np.ndarray, step: float = DWT_STEP) -> np.ndarray:
    _check(cover, bits)
    y = luminance(cover)
    ll, details = _ll4(y)
    rec = _qim(ll, bits.reshape(GRID, GRID).astype(np.float64), step)
    for d in reversed(details):
        rec = haar_idwt2(rec, d)
    return add_luma_offset(cover, rec - y)


def dwt_extract(img: np.ndarray, step: float = DWT_STEP) -> np.ndarray:
    ll, _ = _ll4(luminance(restore_size(img)))
    return _qim_read(ll, step).reshape(-1)
