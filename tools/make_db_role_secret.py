"""Generate the password for the least-privilege `stego_app` database role.

Prints two things, on YOUR terminal only (nothing is written to disk or sent anywhere):
  1. A SCRAM-SHA-256 verifier  -> paste into supabase_hardening.sql in the Supabase SQL editor.
     It is a salted one-way hash, so Supabase's SQL history never contains the real password.
  2. A ready DATABASE_URL line  -> put into .env in place of the owner (postgres) line.

Run from the repo root while .env still holds the OWNER connection string:
    python tools/make_db_role_secret.py
"""
import base64
import hashlib
import hmac
import os
import secrets
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROLE = "stego_app"
ITERATIONS = 4096          # PostgreSQL's default for SCRAM-SHA-256


def scram_verifier(password: str, iterations: int = ITERATIONS) -> str:
    """Same format PostgreSQL stores in pg_authid: SCRAM-SHA-256$iter:salt$StoredKey:ServerKey."""
    salt = secrets.token_bytes(16)
    salted = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    client_key = hmac.new(salted, b"Client Key", hashlib.sha256).digest()
    stored_key = hashlib.sha256(client_key).digest()
    server_key = hmac.new(salted, b"Server Key", hashlib.sha256).digest()
    b64 = lambda b: base64.b64encode(b).decode()
    return f"SCRAM-SHA-256${iterations}:{b64(salt)}${b64(stored_key)}:{b64(server_key)}"


def owner_url_from_env() -> str:
    env = Path(__file__).resolve().parent.parent / ".env"
    url = os.getenv("DATABASE_URL", "")
    if not url and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DATABASE_URL=") and not line.startswith("#"):
                url = line.split("=", 1)[1].split(" #", 1)[0].strip()   # drop inline comments
    if not url.startswith("postgresql"):
        sys.exit("Active DATABASE_URL in .env is not a Postgres URL - put the Supabase owner URL there first.")
    return url


def app_url(owner_url: str, password: str) -> str:
    p = urlsplit(owner_url)
    user = p.username or ""
    if "." not in user:
        sys.exit("Expected a Supabase pooler user like postgres.<project-ref> in DATABASE_URL.")
    project_ref = user.split(".", 1)[1]
    host = p.hostname + (f":{p.port}" if p.port else "")
    return urlunsplit((p.scheme, f"{ROLE}.{project_ref}:{password}@{host}", p.path, p.query, ""))


if __name__ == "__main__":
    pw = secrets.token_hex(24)        # 192 random bits, hex => URL-safe, no encoding needed
    url = app_url(owner_url_from_env(), pw)
    print("\n1) SCRAM verifier - paste over SCRAM_VERIFIER_HERE in the Supabase SQL editor:\n")
    print("   " + scram_verifier(pw))
    print("\n2) New .env line - replace the active DATABASE_URL line with this AFTER the SQL succeeds:\n")
    print("   DATABASE_URL=" + url)
    print("\nKeep this window private. Do not paste either line into chat or commit it.\n")
