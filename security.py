"""Password hashing, JWT access tokens, opaque refresh tokens."""
import hashlib
import secrets
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
