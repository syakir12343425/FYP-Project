"""Reed-Solomon error correction over GF(2^8), pure Python (no dependencies).

Used by payload.py to turn the decoder's ~1% raw bit error rate into exact message recovery.
Code: RS(n, k) with n <= 255 bytes, nsym = n - k parity bytes; corrects up to nsym // 2 wrong
bytes at unknown positions. Field: GF(2^8) with primitive polynomial x^8+x^4+x^3+x^2+1 (0x11d),
generator alpha = 2, first consecutive root alpha^0 (the common "QR code" convention).

Decoding steps (standard textbook pipeline):
  1. syndromes            - all zero  => no errors
  2. Berlekamp-Massey     - error-locator polynomial Lambda(x)
  3. Chien search         - roots of Lambda  => error positions
  4. Forney algorithm     - error magnitudes  => corrected bytes
Polynomials are lists of coefficients, highest degree first.
"""

PRIM = 0x11D

# ---------------------------------------------------------------- GF(2^8) arithmetic
GF_EXP = [0] * 512          # doubled so gf_mul can skip a modulo
GF_LOG = [0] * 256
_x = 1
for _i in range(255):
    GF_EXP[_i] = _x
    GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= PRIM
for _i in range(255, 512):
    GF_EXP[_i] = GF_EXP[_i - 255]


class ReedSolomonError(Exception):
    """Too many errors to correct (or the received word is inconsistent)."""


def gf_mul(x: int, y: int) -> int:
    if x == 0 or y == 0:
        return 0
    return GF_EXP[GF_LOG[x] + GF_LOG[y]]


def gf_div(x: int, y: int) -> int:
    if y == 0:
        raise ZeroDivisionError("GF(2^8) division by zero")
    if x == 0:
        return 0
    return GF_EXP[(GF_LOG[x] + 255 - GF_LOG[y]) % 255]


def gf_pow(x: int, power: int) -> int:
    return GF_EXP[(GF_LOG[x] * power) % 255]


def gf_inverse(x: int) -> int:
    return GF_EXP[255 - GF_LOG[x]]


def gf_poly_scale(p, x):
    return [gf_mul(c, x) for c in p]


def gf_poly_add(p, q):
    r = [0] * max(len(p), len(q))
    for i, c in enumerate(p):
        r[i + len(r) - len(p)] = c
    for i, c in enumerate(q):
        r[i + len(r) - len(q)] ^= c
    return r


def gf_poly_mul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for j, qj in enumerate(q):
        if qj == 0:
            continue
        for i, pi in enumerate(p):
            r[i + j] ^= gf_mul(pi, qj)
    return r


def gf_poly_eval(p, x):
    y = p[0]
    for c in p[1:]:
        y = gf_mul(y, x) ^ c
    return y


# ---------------------------------------------------------------- encoding
def rs_generator_poly(nsym: int):
    g = [1]
    for i in range(nsym):
        g = gf_poly_mul(g, [1, gf_pow(2, i)])
    return g


_GEN_CACHE = {}


def rs_encode(msg, nsym: int) -> list:
    """Systematic encoding: returns msg followed by nsym parity bytes."""
    msg = list(msg)
    if len(msg) + nsym > 255:
        raise ValueError("RS codeword longer than 255 bytes")
    gen = _GEN_CACHE.setdefault(nsym, rs_generator_poly(nsym))
    out = msg + [0] * nsym
    for i in range(len(msg)):          # synthetic division by the generator polynomial
        coef = out[i]
        if coef != 0:
            for j in range(1, len(gen)):
                out[i + j] ^= gf_mul(gen[j], coef)
    return msg + out[len(msg):]


# ---------------------------------------------------------------- decoding
def rs_syndromes(codeword, nsym: int):
    return [gf_poly_eval(codeword, gf_pow(2, i)) for i in range(nsym)]


def _berlekamp_massey(synd, nsym):
    """Error locator Lambda(x), coefficients lowest degree first internally, returned highest-first."""
    err_loc = [1]
    old_loc = [1]
    for i in range(nsym):
        delta = synd[i]
        for j in range(1, len(err_loc)):
            delta ^= gf_mul(err_loc[-(j + 1)], synd[i - j])
        old_loc = old_loc + [0]
        if delta != 0:
            if len(old_loc) > len(err_loc):
                new_loc = gf_poly_scale(old_loc, delta)
                old_loc = gf_poly_scale(err_loc, gf_inverse(delta))
                err_loc = new_loc
            err_loc = gf_poly_add(err_loc, gf_poly_scale(old_loc, delta))
    while len(err_loc) and err_loc[0] == 0:
        del err_loc[0]
    errs = len(err_loc) - 1
    if errs * 2 > nsym:
        raise ReedSolomonError("too many errors to correct")
    return err_loc


def _chien_search(err_loc, nmess):
    """Positions (index into the codeword) where the error locator has roots."""
    errs = len(err_loc) - 1
    positions = []
    for i in range(nmess):
        if gf_poly_eval(err_loc, gf_pow(2, i)) == 0:
            positions.append(nmess - 1 - i)
    if len(positions) != errs:
        raise ReedSolomonError("could not locate all errors (too many errors)")
    return positions


def _forney(codeword, synd, err_pos):
    """Compute error magnitudes and return the corrected codeword."""
    n = len(codeword)
    coef_pos = [n - 1 - p for p in err_pos]
    # errata locator from the known positions
    e_loc = [1]
    for p in coef_pos:
        e_loc = gf_poly_mul(e_loc, gf_poly_add([1], [gf_pow(2, p), 0]))
    # error evaluator Omega(x) = S(x) * Lambda(x) mod x^(len(Lambda)); S is padded with a leading 0
    # (S_0 = 0 term) so that, with fcr = 0, the Forney formula below needs only X_i^1.
    synd_rev = ([0] + synd)[::-1]
    omega = gf_poly_mul(synd_rev, e_loc)
    omega = omega[len(omega) - len(e_loc):]
    x_vals = [gf_pow(2, p) for p in coef_pos]            # X_i = alpha^p
    corrected = list(codeword)
    for i, xi in enumerate(x_vals):
        xi_inv = gf_inverse(xi)
        # formal derivative of the locator evaluated at xi_inv (product form)
        denom = 1
        for j, xj in enumerate(x_vals):
            if j != i:
                denom = gf_mul(denom, 1 ^ gf_mul(xi_inv, xj))
        if denom == 0:
            raise ReedSolomonError("Forney denominator is zero")
        y = gf_poly_eval(omega, xi_inv)
        y = gf_mul(xi, y)                      # fcr = 0  =>  multiply by X_i^(1 - fcr)
        corrected[err_pos[i]] ^= gf_div(y, denom)
    return corrected


def rs_decode(codeword, nsym: int):
    """Returns (message_bytes, number_of_corrected_bytes). Raises ReedSolomonError if uncorrectable."""
    codeword = list(codeword)
    if len(codeword) > 255:
        raise ValueError("RS codeword longer than 255 bytes")
    synd = rs_syndromes(codeword, nsym)
    if max(synd) == 0:
        return codeword[:-nsym], 0
    err_loc = _berlekamp_massey(synd, nsym)
    err_pos = _chien_search(err_loc[::-1], len(codeword))   # Chien works on the reversed locator
    corrected = _forney(codeword, synd, err_pos)
    if max(rs_syndromes(corrected, nsym)) != 0:          # never return an inconsistent "fix"
        raise ReedSolomonError("correction failed verification")
    return corrected[:-nsym], len(err_pos)
