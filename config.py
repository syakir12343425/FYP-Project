"""Central, validated settings. Every secret comes from the environment (.env), never from code."""
import os
from dotenv import load_dotenv

load_dotenv()


def _req(name: str, min_len: int = 1) -> str:
    v = os.getenv(name, "")
    if len(v) < min_len:
        raise RuntimeError(f"Missing/too-short env var {name} (need >= {min_len} chars). See .env.example")
    return v


JWT_SECRET = _req("JWT_SECRET", 32)                 # python -c "import secrets;print(secrets.token_urlsafe(64))"
JWT_ALG = "HS256"
ACCESS_TTL_MIN = int(os.getenv("ACCESS_TTL_MIN", "15"))
REFRESH_TTL_DAYS = int(os.getenv("REFRESH_TTL_DAYS", "7"))

# Supabase (or any Postgres): postgresql+psycopg2://user:pass@host:5432/postgres?sslmode=require
# Falls back to a local SQLite file so you can develop without internet.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stego_auth.db")

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8080").split(",") if o.strip()]
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"   # MUST be true behind HTTPS
ENV = os.getenv("ENV", "dev")                         # "prod" disables /docs

MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15
MIN_PASSWORD_LEN = 12
BCRYPT_ROUNDS = 12

# Two-step verification (TOTP). The key encrypts every user's authenticator secret in the database
# (AES-256-GCM), so a database leak alone does not reveal them. 32 random bytes, base64:
#   python -c "import secrets,base64;print(base64.b64encode(secrets.token_bytes(32)).decode())"
# Changing or losing it invalidates every enrolled authenticator (users would have to set up again).
MFA_ENCRYPTION_KEY = os.getenv("MFA_ENCRYPTION_KEY", "")
REQUIRE_ADMIN_MFA = os.getenv("REQUIRE_ADMIN_MFA", "true").lower() == "true"   # admins must enrol to manage accounts
MFA_ISSUER = "RobustStego"
