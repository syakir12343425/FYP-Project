"""Least-privilege check for the `stego_app` database role (run against Supabase).

Connects with the ACTIVE DATABASE_URL in .env (must be stego_app, the script refuses to run as
the owner) and tries, inside transactions that are always rolled back:
  * operations the app NEEDS        -> must succeed
  * operations an attacker would want -> must be refused (permission denied / must be owner)

Nothing is ever committed, so the database is unchanged afterwards.

    python tests/test_db_privileges.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, text   # noqa: E402
from sqlalchemy.exc import DBAPIError        # noqa: E402

import config                                # noqa: E402

DENIED = {"42501"}   # insufficient_privilege (covers "permission denied" and "must be owner")

MUST_WORK = [
    ("read admin account for login",  "select id, username, password_hash from admin_users limit 1"),
    ("update lockout counter",         "update admin_users set failed_attempts = failed_attempts where id = -1"),
    ("read refresh tokens",            "select count(*) from refresh_tokens"),
    ("revoke a refresh token",         "update refresh_tokens set revoked = true where id = -1"),
    ("read revoked families",          "select count(*) from revoked_families"),
    ("write an audit event",           "insert into audit_log (ts, event) values (now(), 'priv_test') returning id"),
]

MUST_FAIL = [
    ("read audit log contents",        "select event, username, ip from audit_log limit 1"),
    ("edit an audit event",            "update audit_log set detail = 'tampered' where id = -1"),
    ("delete audit events",            "delete from audit_log where id = -1"),
    ("create a new admin",             "insert into admin_users (username, password_hash, failed_attempts) values ('evil', 'x', 0)"),
    ("change an admin password hash",  "update admin_users set password_hash = 'x' where id = -1"),
    ("delete admin accounts",          "delete from admin_users where id = -1"),
    ("delete refresh tokens",          "delete from refresh_tokens where id = -1"),
    ("un-revoke a token family",       "delete from revoked_families where family_id = 'x'"),
    ("drop a table",                   "drop table audit_log"),
    ("disable row level security",     "alter table audit_log disable row level security"),
    ("create a new table",             "create table public.evil (id int)"),
]


def attempt(engine, sql: str) -> tuple[bool, str]:
    """Run sql in a transaction that is ALWAYS rolled back. Returns (succeeded, info)."""
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(text(sql))
            return True, "allowed"
        except DBAPIError as e:
            code = getattr(e.orig, "pgcode", None)
            msg = str(e.orig).strip().splitlines()[0]
            return False, f"[{code}] {msg}" if code not in DENIED else f"refused: {msg}"
        finally:
            trans.rollback()


def main():
    if not config.DATABASE_URL.startswith("postgresql"):
        sys.exit("Active DATABASE_URL is not Postgres - this test is for Supabase.")
    engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
    with engine.connect() as conn:
        who = conn.execute(text("select current_user")).scalar()
    if who != "stego_app":
        sys.exit(f"Connected as '{who}', not 'stego_app'. Switch .env to the stego_app line first.")
    print(f"Connected as: {who}\n")

    failures = 0
    print("Operations the app needs (must be ALLOWED):")
    for name, sql in MUST_WORK:
        ok, info = attempt(engine, sql)
        failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:32s} {info}")

    print("\nOperations an attacker would want (must be REFUSED):")
    for name, sql in MUST_FAIL:
        ok, info = attempt(engine, sql)
        refused = (not ok) and info.startswith("refused")
        failures += not refused
        print(f"  [{'PASS' if refused else 'FAIL'}] {name:32s} {info}")

    total = len(MUST_WORK) + len(MUST_FAIL)
    print(f"\nRESULT: {total - failures}/{total} checks passed" + ("" if failures else " - least privilege holds"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
