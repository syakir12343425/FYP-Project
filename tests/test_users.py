"""Multi-user accounts: roles, admin-created users, forced password change, disable, reset, migration.

Runs against a throwaway SQLite file (never your real database), no PyTorch needed:
    python tests/test_users.py
"""
import os
import sys
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp())
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp / 'users_test.db'}"
os.environ["JWT_SECRET"] = "test-only-secret-" + "x" * 40
os.environ["ALLOWED_ORIGINS"] = "http://localhost:8080"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI               # noqa: E402
from fastapi.testclient import TestClient          # noqa: E402
from sqlalchemy import create_engine, text         # noqa: E402

import admin_routes                                # noqa: E402
import auth_routes                                 # noqa: E402
from db import SessionLocal, User, init_db, migrate_users_schema, missing_user_columns   # noqa: E402
from security import hash_password, password_problem   # noqa: E402

ADMIN, ADMIN_PW = "boss", "Admin-Password-123!"
HDRS = {"X-Requested-With": "fetch"}
results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({info})" if info and not cond else ""))


def make_app() -> FastAPI:
    app = FastAPI()
    app.state.limiter = auth_routes.limiter
    app.include_router(auth_routes.router)
    app.include_router(admin_routes.router)

    @app.get("/api/protected")          # stands in for /api/embed etc. (same dependency)
    def protected(user: User = Depends(auth_routes.require_user)):
        return {"ok": True, "user": user.username}
    return app


def fresh_client(app):
    auth_routes.limiter.reset()
    return TestClient(app)


def login(client, username, password):
    auth_routes.limiter.reset()
    client.cookies.clear()
    return client.post("/api/auth/login", json={"username": username, "password": password}, headers=HDRS)


def bearer(tok):
    return {**HDRS, "Authorization": f"Bearer {tok}"}


def main():
    init_db()
    with SessionLocal() as db:
        db.add(User(username=ADMIN, password_hash=hash_password(ADMIN_PW), role="admin"))
        db.commit()
    app = make_app()
    c = fresh_client(app)

    print("Admin creates an account")
    r = login(c, ADMIN, ADMIN_PW)
    check("admin can log in, role is admin", r.status_code == 200 and r.json()["role"] == "admin", r.text)
    atok = r.json()["access_token"]
    r = c.post("/api/admin/users", json={"username": "alice"}, headers=bearer(atok))
    check("create user -> 201", r.status_code == 201, r.text)
    temp = r.json().get("temporary_password", "")
    alice_id = r.json().get("user", {}).get("id")
    check("temporary password meets the policy", password_problem(temp) is None, temp)
    check("new user must change password", r.json().get("user", {}).get("must_change_password") is True)
    r = c.post("/api/admin/users", json={"username": "ALICE"}, headers=bearer(atok))
    check("duplicate username (any case) -> 409", r.status_code == 409, r.text)
    r = c.post("/api/admin/users", json={"username": "<script>"}, headers=bearer(atok))
    check("bad username -> 400 or 422", r.status_code in (400, 422), r.text)
    r = c.get("/api/admin/users", headers=bearer(atok))
    check("list users shows both, never a password hash",
          r.status_code == 200 and len(r.json()["users"]) == 2 and "password_hash" not in r.text, r.text)

    print("First login with the temporary password")
    u = fresh_client(app)
    r = login(u, "alice", temp)
    check("login works, must_change_password true", r.status_code == 200 and r.json()["must_change_password"], r.text)
    old_cookie = r.cookies.get("refresh_token")
    utok = r.json()["access_token"]
    check("app endpoints refused until password changed (403)", u.get("/api/protected", headers=bearer(utok)).status_code == 403)
    check("/me still works", u.get("/api/auth/me", headers=bearer(utok)).status_code == 200)
    r = u.post("/api/auth/change-password", json={"current_password": "wrong", "new_password": "Brand-New-Pass-42!"}, headers=bearer(utok))
    check("wrong current password -> 400", r.status_code == 400, r.text)
    auth_routes.limiter.reset()
    r = u.post("/api/auth/change-password", json={"current_password": temp, "new_password": "short"}, headers=bearer(utok))
    check("weak new password -> 400", r.status_code == 400, r.text)
    auth_routes.limiter.reset()
    r = u.post("/api/auth/change-password", json={"current_password": temp, "new_password": "Alice-Password-99!"}, headers=bearer(utok))
    check("username inside password -> 400", r.status_code == 400, r.text)
    auth_routes.limiter.reset()
    r = u.post("/api/auth/change-password", json={"current_password": temp, "new_password": "Brand-New-Pass-42!"}, headers=bearer(utok))
    check("good new password -> 200, flag cleared", r.status_code == 200 and not r.json()["must_change_password"], r.text)
    utok = r.json()["access_token"]
    check("app endpoints now allowed", u.get("/api/protected", headers=bearer(utok)).status_code == 200)
    x = TestClient(app)
    r = x.post("/api/auth/refresh", headers={**HDRS, "Cookie": f"refresh_token={old_cookie}"})
    check("session from before the change is revoked", r.status_code == 401, r.text)
    check("temporary password no longer works", login(fresh_client(app), "alice", temp).status_code == 401)

    print("Normal users cannot manage accounts")
    check("user -> /api/admin/users is 403", u.get("/api/admin/users", headers=bearer(utok)).status_code == 403)
    check("user cannot create accounts", u.post("/api/admin/users", json={"username": "mallory"}, headers=bearer(utok)).status_code == 403)

    print("Disable / enable")
    r = c.post(f"/api/admin/users/{alice_id}/disable", headers=bearer(atok))
    check("disable -> 200", r.status_code == 200 and r.json()["user"]["disabled"], r.text)
    check("disabled user's live token stops at once (401)", u.get("/api/protected", headers=bearer(utok)).status_code == 401)
    check("disabled user cannot log in", login(fresh_client(app), "alice", "Brand-New-Pass-42!").status_code == 401)
    c.post(f"/api/admin/users/{alice_id}/enable", headers=bearer(atok))
    check("enabled again -> can log in", login(fresh_client(app), "alice", "Brand-New-Pass-42!").status_code == 200)

    print("Admin resets a forgotten password")
    r = c.post(f"/api/admin/users/{alice_id}/reset-password", headers=bearer(atok))
    temp2 = r.json().get("temporary_password", "")
    check("reset -> new temporary password", r.status_code == 200 and temp2 and temp2 != temp, r.text)
    check("old password no longer works", login(fresh_client(app), "alice", "Brand-New-Pass-42!").status_code == 401)
    r = login(fresh_client(app), "alice", temp2)
    check("temporary password works and forces a change", r.status_code == 200 and r.json()["must_change_password"], r.text)

    print("Admin accounts are out of reach of the app")
    with SessionLocal() as db:
        admin_id = db.query(User).filter_by(username=ADMIN).one().id
    check("cannot reset an admin in the app", c.post(f"/api/admin/users/{admin_id}/reset-password", headers=bearer(atok)).status_code == 403)
    check("cannot disable an admin in the app", c.post(f"/api/admin/users/{admin_id}/disable", headers=bearer(atok)).status_code == 403)
    auth_routes.limiter.reset()
    r = c.post("/api/auth/change-password", json={"current_password": ADMIN_PW, "new_password": "Other-Admin-Pass-77!"}, headers=bearer(atok))
    check("admin password not changeable in the app", r.status_code == 403, r.text)

    print("Migration of an old (single-admin) database")
    old = create_engine(f"sqlite:///{_tmp / 'old_schema.db'}")
    with old.begin() as conn:
        conn.execute(text("CREATE TABLE admin_users (id INTEGER PRIMARY KEY, username VARCHAR(64) UNIQUE, "
                          "password_hash VARCHAR(128), failed_attempts INTEGER, locked_until DATETIME, created_at DATETIME)"))
        conn.execute(text("INSERT INTO admin_users (username, password_hash, failed_attempts) VALUES ('oldadmin', 'x', 0)"))
    added = migrate_users_schema(old)
    with old.connect() as conn:
        role = conn.execute(text("SELECT role FROM admin_users WHERE username = 'oldadmin'")).scalar()
    check("adds the 4 new columns", sorted(added) == sorted(["role", "disabled", "must_change_password", "last_login_at"]), added)
    check("existing account becomes admin", role == "admin", role)
    check("re-running does nothing", migrate_users_schema(old) == [] and missing_user_columns(old) == [])

    print(f"\nRESULT: {sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
