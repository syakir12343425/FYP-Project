"""Two-step verification (TOTP): set-up, sign-in, replay, recovery codes, lockout, admin rules.

Throwaway SQLite database, no PyTorch, no phone needed (the test computes the codes itself):
    python tests/test_mfa.py
"""
import base64
import os
import secrets
import sys
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp())
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp / 'mfa_test.db'}"
os.environ["JWT_SECRET"] = "test-only-secret-" + "x" * 40
os.environ["ALLOWED_ORIGINS"] = "http://localhost:8080"
os.environ["REQUIRE_MFA_ALL"] = "false"            # compulsory two-step is tested in tests/test_signup.py
os.environ["MFA_ENCRYPTION_KEY"] = base64.b64encode(secrets.token_bytes(32)).decode()
os.environ["REQUIRE_ADMIN_MFA"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI              # noqa: E402
from fastapi.testclient import TestClient         # noqa: E402

import admin_routes                               # noqa: E402
import auth_routes                                # noqa: E402
import security                                   # noqa: E402
from db import MfaRecoveryCode, SessionLocal, User, init_db   # noqa: E402
from security import hash_password                # noqa: E402

HDRS = {"X-Requested-With": "fetch"}
ADMIN, ADMIN_PW = "boss", "Admin-Password-123!"
USER, USER_PW = "dina", "Dina-Strong-Pass-55!"
results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({info})" if info and not cond else ""))


def make_app():
    app = FastAPI()
    app.state.limiter = auth_routes.limiter
    app.include_router(auth_routes.router)
    app.include_router(admin_routes.router)

    @app.get("/api/protected")
    def protected(user: User = Depends(auth_routes.require_user)):
        return {"ok": True}
    return app


def post(c, path, body=None, tok=None):
    auth_routes.limiter.reset()
    h = dict(HDRS)
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return c.post(path, json=body, headers=h)


def code_for(secret, offset=0):
    return security._hotp(secret, security.totp_now_step() + offset)


def enrol(c, tok, password):
    r = post(c, "/api/auth/mfa/setup", {"password": password}, tok)
    secret = r.json().get("secret")
    r2 = post(c, "/api/auth/mfa/enable", {"code": code_for(secret)}, tok)
    return secret, r, r2


def main():
    init_db()
    with SessionLocal() as db:
        db.add(User(username=ADMIN, password_hash=hash_password(ADMIN_PW), role="admin"))
        db.add(User(username=USER, password_hash=hash_password(USER_PW), role="user"))
        db.commit()
    app = make_app()
    c = TestClient(app)

    print("Admins must turn on two-step verification before managing accounts")
    r = post(c, "/api/auth/login", {"username": ADMIN, "password": ADMIN_PW})
    atok = r.json()["access_token"]
    check("admin without MFA gets 403 on /api/admin/users", c.get("/api/admin/users", headers={"Authorization": f"Bearer {atok}"}).status_code == 403)

    print("Set-up")
    r = post(c, "/api/auth/mfa/setup", {"password": "wrong"}, atok)
    check("set-up needs the password", r.status_code == 400, r.text)
    r = post(c, "/api/auth/mfa/setup", {"password": ADMIN_PW}, atok)
    secret = r.json().get("secret", "")
    check("set-up returns a secret and an otpauth link", r.status_code == 200 and r.json()["otpauth_uri"].startswith("otpauth://totp/"), r.text)
    with SessionLocal() as db:
        row = db.query(User).filter_by(username=ADMIN).one()
        check("secret is stored encrypted, not in clear", secret and secret not in (row.mfa_pending_enc or "") and row.mfa_pending_enc.startswith("v1:"))
    r = post(c, "/api/auth/mfa/enable", {"code": "000000"}, atok)
    check("wrong code does not enable", r.status_code == 400, r.text)
    r = post(c, "/api/auth/mfa/enable", {"code": code_for(secret)}, atok)
    rc = r.json().get("recovery_codes", [])
    check("right code enables, 10 recovery codes shown once", r.status_code == 200 and len(rc) == 10, r.text)
    with SessionLocal() as db:
        hashes = [x.code_hash for x in db.query(MfaRecoveryCode).all()]
        check("recovery codes stored only as hashes", len(hashes) == 10 and not any(code in hashes for code in rc))
    check("admin can now manage accounts", c.get("/api/admin/users", headers={"Authorization": f"Bearer {atok}"}).status_code == 200)

    print("Signing in with two-step verification")
    c2 = TestClient(app)
    r = post(c2, "/api/auth/login", {"username": ADMIN, "password": ADMIN_PW})
    check("password step returns mfa_required, no session yet",
          r.status_code == 200 and r.json().get("mfa_required") and "access_token" not in r.json() and "refresh_token" not in r.cookies, r.text)
    mtok = r.json()["mfa_token"]
    check("the mfa token is not accepted as an access token",
          c2.get("/api/protected", headers={"Authorization": f"Bearer {mtok}"}).status_code == 401)
    code = code_for(secret)
    with SessionLocal() as db:            # the enable step used the current step; move it back so this code is fresh
        db.query(User).filter_by(username=ADMIN).update({"mfa_last_step": security.totp_now_step() - 1})
        db.commit()
    r = post(c2, "/api/auth/mfa", {"mfa_token": mtok, "code": code})
    check("right code -> session", r.status_code == 200 and "access_token" in r.json(), r.text)
    r = post(TestClient(app), "/api/auth/mfa", {"mfa_token": mtok, "code": code})
    check("same code cannot be used twice (replay)", r.status_code == 401, r.text)
    r = post(c2, "/api/auth/mfa", {"mfa_token": "not-a-token-at-all", "code": code})
    check("forged mfa token refused", r.status_code == 401)

    print("Recovery codes")
    r = post(c2, "/api/auth/login", {"username": ADMIN, "password": ADMIN_PW})
    mtok = r.json()["mfa_token"]
    r = post(c2, "/api/auth/mfa", {"mfa_token": mtok, "code": rc[0].upper().replace("-", " ")})
    check("recovery code works (any case, spaces ok)", r.status_code == 200, r.text)
    r = post(c2, "/api/auth/login", {"username": ADMIN, "password": ADMIN_PW})
    r = post(c2, "/api/auth/mfa", {"mfa_token": r.json()["mfa_token"], "code": rc[0]})
    check("a recovery code works only once", r.status_code == 401, r.text)
    st = c2.get("/api/auth/mfa/status", headers={"Authorization": f"Bearer {atok}"}).json()
    check("status shows 9 recovery codes left", st.get("recovery_codes_left") == 9, st)

    print("Lockout counts wrong codes")
    with SessionLocal() as db:
        db.query(User).filter_by(username=ADMIN).update({"failed_attempts": 0, "locked_until": None})
        db.commit()
    for _ in range(5):
        r = post(c2, "/api/auth/login", {"username": ADMIN, "password": ADMIN_PW})
        if r.json().get("mfa_token"):
            post(c2, "/api/auth/mfa", {"mfa_token": r.json()["mfa_token"], "code": "123456"})
    r = post(c2, "/api/auth/login", {"username": ADMIN, "password": ADMIN_PW})
    check("5 wrong codes lock the account", r.status_code == 401, r.text)
    with SessionLocal() as db:
        db.query(User).filter_by(username=ADMIN).update({"failed_attempts": 0, "locked_until": None})
        db.commit()

    print("Admin rules")
    r = post(c, "/api/auth/mfa/disable", {"password": ADMIN_PW, "code": code_for(secret, 1)}, atok)
    check("admin cannot turn two-step off in the app", r.status_code == 403, r.text)

    print("Normal user: optional, can turn off, admin can reset after a lost phone")
    cu = TestClient(app)
    r = post(cu, "/api/auth/login", {"username": USER, "password": USER_PW})
    utok = r.json()["access_token"]
    usecret, _, r2 = enrol(cu, utok, USER_PW)
    check("user enrols", r2.status_code == 200, r2.text)
    r = post(cu, "/api/auth/mfa/disable", {"password": USER_PW, "code": "000000"}, utok)
    check("turning off needs a valid code", r.status_code == 400, r.text)
    with SessionLocal() as db:
        uid = db.query(User).filter_by(username=USER).one().id
    r = c.get("/api/admin/users", headers={"Authorization": f"Bearer {atok}"})
    row = [u for u in r.json()["users"] if u["username"] == USER][0]
    check("admin list shows the user's two-step status", row["mfa_enabled"] is True)
    r = post(c, f"/api/admin/users/{uid}/reset-mfa", None, atok)
    check("admin resets the user's two-step (lost phone)", r.status_code == 200 and r.json()["user"]["mfa_enabled"] is False, r.text)
    r = post(TestClient(app), "/api/auth/login", {"username": USER, "password": USER_PW})
    check("user signs in with password only after the reset", r.status_code == 200 and "access_token" in r.json(), r.text)
    with SessionLocal() as db:
        admin_id = db.query(User).filter_by(username=ADMIN).one().id
    check("admin's own two-step cannot be reset in the app", post(c, f"/api/admin/users/{admin_id}/reset-mfa", None, atok).status_code == 403)

    print(f"\nRESULT: {sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
