"""Self sign-up with email confirmation and compulsory two-step verification.

Throwaway SQLite database, no PyTorch, no real email (messages are captured in memory):
    python tests/test_signup.py
"""
import os
import re
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

_tmp = Path(tempfile.mkdtemp())
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp / 'signup_test.db'}"
os.environ["JWT_SECRET"] = "test-only-secret-" + "x" * 40
os.environ["ALLOWED_ORIGINS"] = "http://localhost:8080"
os.environ["MFA_ENCRYPTION_KEY"] = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU="
os.environ["REQUIRE_MFA_ALL"] = "true"
os.environ["SIGNUP_REQUIRES_APPROVAL"] = "false"
os.environ["MAX_PENDING_SIGNUPS"] = "6"
os.environ["PUBLIC_URL"] = "https://stego.example"
os.environ["ENV"] = "dev"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI              # noqa: E402
from fastapi.testclient import TestClient         # noqa: E402

import admin_routes                               # noqa: E402
import auth_routes                                # noqa: E402
import security                                   # noqa: E402
from db import SessionLocal, User, init_db, utcnow   # noqa: E402
from security import hash_password                # noqa: E402

HDRS = {"X-Requested-With": "fetch"}
PW = "Signup-Password-77!"
OUTBOX = []
auth_routes.send_mail_quietly = lambda to, subject, body: OUTBOX.append((to, subject, body))
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


def post(c, path, body=None, tok=None, headers=None):
    auth_routes.limiter.reset()
    h = dict(HDRS if headers is None else headers)
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return c.post(path, json=body, headers=h)


def get(c, path, tok):
    return c.get(path, headers={"Authorization": f"Bearer {tok}"})


def register(c, email, username, password=PW):
    return post(c, "/api/auth/register", {"email": email, "username": username, "password": password})


def last_token():
    m = re.search(r"#verify=([A-Za-z0-9_-]+)", OUTBOX[-1][2])
    return m.group(1) if m else ""


def main():
    init_db()
    app = make_app()
    c = TestClient(app)

    print("Signing up")
    r = register(c, "Nadia@Example.com", "nadia")
    check("sign-up -> 201 'check your email'", r.status_code == 201 and "email" in r.text, r.text)
    to, subject, body = OUTBOX[-1]
    token = last_token()
    check("confirmation email sent to the address (lower-cased), link uses PUBLIC_URL and #fragment",
          to == "nadia@example.com" and "https://stego.example/#verify=" in body and token, OUTBOX[-1])
    with SessionLocal() as db:
        u = db.query(User).filter_by(username="nadia").one()
        check("account is a normal user, email not yet confirmed", u.role == "user" and not u.email_verified)
        check("only a hash of the link token is stored", u.email_token_hash and token not in (u.email_token_hash or ""))
    check("bad email refused", register(c, "not-an-email", "someone1").status_code == 400)
    check("weak password refused", register(c, "w@example.com", "weakling", "password").status_code == 400)
    check("bad username refused", register(c, "b@example.com", "<b>x</b>").status_code == 400)
    check("taken username refused", register(c, "other@example.com", "NADIA").status_code == 409)
    check("no CSRF header -> refused", post(c, "/api/auth/register", {"email": "z@example.com", "username": "zed00", "password": PW}, headers={}).status_code == 403)
    r = post(c, "/api/auth/register", {"email": "s@example.com", "username": "sneaky", "password": PW, "role": "admin", "email_verified": True})
    with SessionLocal() as db:
        s = db.query(User).filter_by(username="sneaky").one_or_none()
        check("extra fields (role, email_verified) are ignored", s is not None and s.role == "user" and not s.email_verified, r.text)

    print("Same email again (no account enumeration)")
    n = len(OUTBOX)
    r = register(c, "NADIA@example.com", "nadia2")
    with SessionLocal() as db:
        none = db.query(User).filter_by(username="nadia2").one_or_none()
    check("same answer as a new sign-up, no second account", r.status_code == 201 and none is None, r.text)
    check("the owner gets a heads-up email instead of a link", len(OUTBOX) == n + 1 and "#verify=" not in OUTBOX[-1][2])

    print("Before confirming")
    r = post(TestClient(app), "/api/auth/login", {"username": "nadia", "password": PW})
    check("right password -> 403 'confirm your email', masked address, no session",
          r.status_code == 403 and "n***@example.com" in r.text and "refresh_token" not in r.cookies, r.text)
    check("wrong password -> generic 401", post(TestClient(app), "/api/auth/login", {"username": "nadia", "password": "Wrong-Password-1!"}).status_code == 401)

    print("New link")
    check("resend needs the password", post(c, "/api/auth/resend-verification", {"username": "nadia", "password": "Wrong-Password-1!"}).status_code == 401)
    r = post(c, "/api/auth/resend-verification", {"username": "nadia", "password": PW})
    new_token = last_token()
    check("resend -> new link emailed", r.status_code == 200 and new_token and new_token != token, r.text)
    check("the old link stops working", post(c, "/api/auth/verify-email", {"token": token}).status_code == 400)

    print("Confirming")
    check("garbage token refused", post(c, "/api/auth/verify-email", {"token": "x" * 43}).status_code == 400)
    r = post(c, "/api/auth/verify-email", {"token": new_token})
    check("link confirms the email", r.status_code == 200, r.text)
    check("the link works only once", post(c, "/api/auth/verify-email", {"token": new_token}).status_code == 400)

    print("First sign-in: two-step verification is compulsory")
    cu = TestClient(app)
    r = post(cu, "/api/auth/login", {"username": "nadia", "password": PW})
    utok = r.json().get("access_token")
    check("sign-in works, response says two-step is required", r.status_code == 200 and r.json()["mfa_required"] is True, r.text)
    r = get(cu, "/api/protected", utok)
    check("app refused until two-step is on (403)", r.status_code == 403 and "two-step" in r.text, r.text)
    check("but the two-step status/set-up pages work", get(cu, "/api/auth/mfa/status", utok).status_code == 200)
    r = post(cu, "/api/auth/mfa/setup", {"password": PW}, utok)
    secret = r.json().get("secret", "")
    r = post(cu, "/api/auth/mfa/enable", {"code": security._hotp(secret, security.totp_now_step())}, utok)
    check("set up two-step", r.status_code == 200, r.text)
    check("now the app works", get(cu, "/api/protected", utok).status_code == 200)
    r = post(cu, "/api/auth/mfa/disable", {"password": PW, "code": "000000"}, utok)
    check("two-step cannot be turned off when it is compulsory", r.status_code == 403, r.text)

    print("Address held by someone who never confirmed")
    register(c, "victim@example.com", "squatter")
    with SessionLocal() as db:
        db.query(User).filter_by(username="squatter").update({"email_token_expires": utcnow() - timedelta(hours=1)})
        db.commit()
    r = register(c, "victim@example.com", "realowner")
    with SessionLocal() as db:
        sq = db.query(User).filter_by(username="squatter").one()
        ro = db.query(User).filter_by(username="realowner").one_or_none()
    check("expired unconfirmed sign-up releases the address to the real owner",
          r.status_code == 201 and ro is not None and ro.email == "victim@example.com" and sq.email is None and sq.disabled, r.text)

    print("Accounts made by an admin or create_admin.py need no email")
    with SessionLocal() as db:
        db.add(User(username="madebyadmin", password_hash=hash_password(PW), role="user"))
        db.commit()
    r = post(TestClient(app), "/api/auth/login", {"username": "madebyadmin", "password": PW})
    check("signs in without an email step", r.status_code == 200, r.text)

    print("Abuse limits")
    for i in range(6):
        register(c, f"bot{i}@example.com", f"bot{i:03d}")
    r = register(c, "bot99@example.com", "bot099")
    check("sign-ups pause while too many are unconfirmed (503)", r.status_code == 503, r.text)
    auth_routes.limiter.reset()
    codes = [c.post("/api/auth/register", json={"email": f"r{i}@example.com", "username": f"rate{i}", "password": PW},
                    headers=HDRS).status_code for i in range(5)]
    check("more than 3 sign-ups an hour from one IP -> 429", 429 in codes, codes)

    print(f"\nRESULT: {sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
