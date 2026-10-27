"""One-off CLI: create/reset the single admin account.   python create_admin.py"""
import getpass
import re
import sys

import config
from db import AdminUser, SessionLocal, init_db
from security import hash_password

init_db()
username = input("Admin username: ").strip()
pw = getpass.getpass(f"Password (min {config.MIN_PASSWORD_LEN} chars): ")
if pw != getpass.getpass("Repeat password: "):
    sys.exit("Passwords differ.")
if (len(pw) < config.MIN_PASSWORD_LEN or not re.search(r"[a-z]", pw) or not re.search(r"[A-Z]", pw)
        or not re.search(r"\d", pw) or not re.search(r"[^\w\s]", pw)):
    sys.exit("Weak password: need length, upper, lower, digit and symbol.")
with SessionLocal() as db:
    u = db.query(AdminUser).filter_by(username=username).first()
    if u:
        u.password_hash, u.failed_attempts, u.locked_until = hash_password(pw), 0, None
        print("Existing admin password reset.")
    else:
        db.add(AdminUser(username=username, password_hash=hash_password(pw)))
        print("Admin created.")
    db.commit()
