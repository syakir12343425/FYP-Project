"""One-off CLI: create an ADMIN account, or reset an admin's password.   python create_admin.py
Needs the database OWNER url in .env when using Supabase: the server's own role (stego_app) is not
allowed to create admins or change their passwords (enforced by a trigger in supabase_hardening.sql).
Normal user accounts are created by an admin inside the app (Users page)."""
import getpass
import sys

from db import MfaRecoveryCode, SessionLocal, User, init_db, utcnow
from security import hash_password, password_problem

init_db()
username = input("Admin username: ").strip()
pw = getpass.getpass("Password (min 12 chars, upper, lower, digit, symbol): ")
if pw != getpass.getpass("Repeat password: "):
    sys.exit("Passwords differ.")
problem = password_problem(pw, username)
if problem:
    sys.exit(f"Weak password: {problem}")
with SessionLocal() as db:
    u = db.query(User).filter_by(username=username).first()
    if u and u.role != "admin":
        sys.exit(f"'{username}' is a normal user. Reset it from the Users page instead.")
    if u:
        u.password_hash, u.failed_attempts, u.locked_until = hash_password(pw), 0, None
        u.disabled, u.must_change_password = False, False
        print("Existing admin password reset.")
        if u.mfa_enabled and input("Also turn off two-step verification (lost phone)? [y/N] ").strip().lower() == "y":
            u.mfa_enabled, u.mfa_secret_enc, u.mfa_pending_enc, u.mfa_pending_at = False, None, None, None
            db.query(MfaRecoveryCode).filter(MfaRecoveryCode.user_id == u.id,
                                             MfaRecoveryCode.used_at.is_(None)).update({"used_at": utcnow()})
            print("Two-step verification turned off. Set it up again in the app (Account page).")
    else:
        db.add(User(username=username, password_hash=hash_password(pw), role="admin"))
        print("Admin created.")
    db.commit()
