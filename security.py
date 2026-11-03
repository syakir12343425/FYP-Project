"""Password hashing, JWT access tokens, opaque refresh tokens."""
import hashlib
import re
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
