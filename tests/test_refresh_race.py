"""Race test for refresh-token rotation.

Two requests replay the SAME refresh cookie at the same moment (what an attacker holding a stolen
cookie would try). Safe behaviour: at most one of them gets a new session, and the reuse is
detected, so the whole token family is revoked.

Run from the repo root (uses a throwaway SQLite file, never your real database):
    python tests/test_refresh_race.py
"""
import os
import sys
import tempfile
import threading
from pathlib import Path

# Configure BEFORE importing the app modules (config.py reads env vars at import time).
_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_tmp) / 'race_test.db'}"
os.environ["JWT_SECRET"] = "test-only-secret-" + "x" * 40
os.environ["ALLOWED_ORIGINS"] = "http://localhost:8080"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI                      # noqa: E402
from fastapi.testclient import TestClient        # noqa: E402

import auth_routes                               # noqa: E402
from db import AdminUser, SessionLocal, init_db  # noqa: E402
from security import hash_password               # noqa: E402

USER, PW = "race_admin", "Test-Password-123!"
HDRS = {"X-Requested-With": "fetch"}


def make_app() -> FastAPI:
    app = FastAPI()
    app.state.limiter = auth_routes.limiter
    app.include_router(auth_routes.router)
    return app


def login(client: TestClient) -> str:
    r = client.post("/api/auth/login", json={"username": USER, "password": PW}, headers=HDRS)
    assert r.status_code == 200, f"login failed: {r.status_code} {r.text}"
    return r.cookies["refresh_token"]


def refresh_with(client: TestClient, cookie: str):
    # Cookie sent explicitly (jar cleared) so each call replays exactly this token.
    client.cookies.clear()
    return client.post("/api/auth/refresh", headers={**HDRS, "Cookie": f"refresh_token={cookie}"})


def main(rounds: int = 50):
    init_db()
    with SessionLocal() as db:
        db.add(AdminUser(username=USER, password_hash=hash_password(PW)))
        db.commit()

    client = TestClient(make_app())
    unsafe = 0
    for i in range(rounds):
        if i and i % 4 == 0:
            auth_routes.limiter.reset()          # stay under the 5/min login limit
        cookie = login(client)
        client.cookies.clear()

        barrier = threading.Barrier(2)
        results = []

        def worker():
            barrier.wait()                       # release both requests together
            results.append(refresh_with(client, cookie))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        codes = sorted(r.status_code for r in results)
        winners = [r for r in results if r.status_code == 200]
        if len(winners) > 1:
            unsafe += 1
            print(f"round {i}: UNSAFE - both replays got a new session {codes}")
            continue
        # If one request won, its brand-new token must ALSO be dead (family revoked on reuse).
        if winners:
            new_cookie = winners[0].cookies["refresh_token"]
            after = refresh_with(client, new_cookie).status_code
            assert after == 401, f"round {i}: family not revoked (got {after})"
        print(f"round {i}: ok {codes}")

    print("\nRESULT:", "PASS - no double rotation" if unsafe == 0 else f"FAIL - {unsafe}/{rounds} unsafe rounds")
    sys.exit(1 if unsafe else 0)


if __name__ == "__main__":
    main()
