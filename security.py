"""Password hashing, JWT access tokens, opaque refresh tokens."""
import base64
import hashlib
import hmac
import re
import struct
import time
from urllib.parse import quote
import secrets
import string
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
import config


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode()[:72], bcrypt.gensalt(config.BCRYPT_ROUNDS)).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode()[:72], hashed.encode())
    except ValueError:
        return False


# Verified against when the username doesn't exist, so response time doesn't reveal valid usernames.
DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def create_access_token(user_id: int, username: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "usr": username, "typ": "access", "iat": now,
               "exp": now + timedelta(minutes=config.ACCESS_TTL_MIN), "jti": secrets.token_hex(8)}
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALG)


def decode_access_token(token: str) -> dict:
    """Raises jwt.PyJWTError on bad signature / expiry. Algorithm is pinned (no 'alg: none' attacks)."""
    data = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALG], options={"require": ["exp", "sub", "iat"]})
    if data.get("typ") != "access":
        raise jwt.InvalidTokenError("wrong token type")
    return data


MFA_TOKEN_TTL_SEC = 300


def create_mfa_token(user_id: int) -> str:
    """Proof that the password was right, valid 5 minutes, good ONLY for /api/auth/mfa (typ 'mfa'),
    never accepted as an access token (decode_access_token insists on typ 'access')."""
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "typ": "mfa", "iat": now, "exp": now + timedelta(seconds=MFA_TOKEN_TTL_SEC),
               "jti": secrets.token_hex(8)}
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALG)


def decode_mfa_token(token: str) -> dict:
    data = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALG], options={"require": ["exp", "sub", "iat"]})
    if data.get("typ") != "mfa":
        raise jwt.InvalidTokenError("wrong token type")
    return data


def qr_svg(text: str) -> str | None:
    """QR code as SVG text for the set-up screen (needs `pip install segno`). None if segno is missing:
    the UI then shows the secret for typing into the app by hand."""
    try:
        import segno
        return segno.make(text, error="m").svg_inline(scale=5, border=2, dark="#000000", light="#ffffff")
    except Exception:
        return None


def new_refresh_token() -> tuple[str, str]:
    raw = secrets.token_urlsafe(48)
    return raw, hash_refresh_token(raw)


def hash_refresh_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()   # DB leak does not yield usable tokens


def password_problem(pw: str, username: str = "") -> str | None:
    """The one password policy (create_admin.py, change-password). None = acceptable."""
    if len(pw) < config.MIN_PASSWORD_LEN:
        return f"Use at least {config.MIN_PASSWORD_LEN} characters."
    if len(pw.encode()) > 72:
        return "Use at most 72 bytes (bcrypt ignores anything longer)."
    if not (re.search(r"[a-z]", pw) and re.search(r"[A-Z]", pw) and re.search(r"\d", pw) and re.search(r"[^\w\s]", pw)):
        return "Mix upper case, lower case, a digit and a symbol."
    if username and username.lower() in pw.lower():
        return "Do not put your username in the password."
    return None


_TEMP_ALPHABET = "".join(c for c in string.ascii_letters + string.digits if c not in "Il1O0o")
_TEMP_SYMBOLS = "-_!?#%+="


def temporary_password(length: int = 16) -> str:
    """Random one-time password an admin hands to a user. Meets password_problem() by construction."""
    while True:
        chars = [secrets.choice(_TEMP_ALPHABET) for _ in range(length - 2)] + [secrets.choice(_TEMP_SYMBOLS) for _ in range(2)]
        # shuffle with a CSPRNG
        for i in range(len(chars) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            chars[i], chars[j] = chars[j], chars[i]
        pw = "".join(chars)
        if password_problem(pw) is None:
            return pw


# ------------------------------------------------------------------ two-step verification (TOTP)
TOTP_PERIOD, TOTP_DIGITS, TOTP_WINDOW = 30, 6, 1     # RFC 6238 defaults; accept 1 step of clock drift


def totp_new_secret() -> str:
    """160-bit random secret, base32 without padding (what authenticator apps expect)."""
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def _hotp(secret_b32: str, counter: int, digits: int = TOTP_DIGITS) -> str:
    key = base64.b32decode(secret_b32 + "=" * (-len(secret_b32) % 8), casefold=True)
    mac = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    off = mac[-1] & 0x0F
    value = struct.unpack(">I", mac[off:off + 4])[0] & 0x7FFFFFFF
    return str(value % 10 ** digits).zfill(digits)


def totp_now_step(now: float | None = None) -> int:
    return int((time.time() if now is None else now) // TOTP_PERIOD)


def totp_verify(secret_b32: str, code: str, last_step: int, now: float | None = None) -> int | None:
    """Return the time step the code belongs to, or None. A step at or before last_step is refused,
    so a code that was already used (or an older one) can never be replayed."""
    code = (code or "").strip().replace(" ", "")
    if not (code.isdigit() and len(code) == TOTP_DIGITS):
        return None
    step = totp_now_step(now)
    for s in range(step - TOTP_WINDOW, step + TOTP_WINDOW + 1):
        if s > last_step and hmac.compare_digest(_hotp(secret_b32, s), code):
            return s
    return None


def otpauth_uri(secret_b32: str, username: str) -> str:
    issuer = quote(config.MFA_ISSUER)
    return (f"otpauth://totp/{issuer}:{quote(username)}?secret={secret_b32}&issuer={issuer}"
            f"&algorithm=SHA1&digits={TOTP_DIGITS}&period={TOTP_PERIOD}")


def _mfa_key() -> bytes:
    raw = config.MFA_ENCRYPTION_KEY
    try:
        key = base64.b64decode(raw, validate=True)
    except Exception:
        key = b""
    if len(key) != 32:
        raise RuntimeError("MFA_ENCRYPTION_KEY must be 32 random bytes in base64. See .env.example")
    return key


def encrypt_mfa_secret(secret_b32: str, user_id: int) -> str:
    """AES-256-GCM. The user id is bound in as associated data, so an encrypted secret copied to another
    account's row does not decrypt."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = secrets.token_bytes(12)
    ct = AESGCM(_mfa_key()).encrypt(nonce, secret_b32.encode(), f"user:{user_id}".encode())
    return "v1:" + base64.b64encode(nonce + ct).decode()


def decrypt_mfa_secret(blob: str, user_id: int) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if not blob or not blob.startswith("v1:"):
        raise ValueError("unknown secret format")
    raw = base64.b64decode(blob[3:])
    return AESGCM(_mfa_key()).decrypt(raw[:12], raw[12:], f"user:{user_id}".encode()).decode()


_RECOVERY_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"     # no 0/o, 1/l/i


def new_recovery_codes(n: int = 10) -> list[str]:
    """10 one-time codes like 'k7m2q-9xwpr' (50 bits each)."""
    return ["".join(secrets.choice(_RECOVERY_ALPHABET) for _ in range(5)) + "-" +
            "".join(secrets.choice(_RECOVERY_ALPHABET) for _ in range(5)) for _ in range(n)]


def normalise_recovery_code(code: str) -> str:
    c = re.sub(r"[^a-z0-9]", "", (code or "").lower())
    return f"{c[:5]}-{c[5:]}" if len(c) == 10 else ""


def hash_recovery_code(code: str) -> str:
    """Keyed hash (HMAC with a key derived from MFA_ENCRYPTION_KEY): a database leak alone cannot be
    brute-forced offline, because the key is not in the database."""
    pepper = hmac.new(_mfa_key(), b"recovery-code-pepper", hashlib.sha256).digest()
    return hmac.new(pepper, normalise_recovery_code(code).encode(), hashlib.sha256).hexdigest()
