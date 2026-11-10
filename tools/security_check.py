"""Black-box security check of a running RobustStego server (your own server only).

    python tools/security_check.py https://<laptop>.<tailnet>.ts.net
    python tools/security_check.py http://localhost:8000

What it does, without ever signing in and without touching real accounts:
  1. Security headers on the web app and the API (CSP, HSTS, X-Frame-Options, ...), no Server banner
  2. Closed doors: /docs, /openapi.json, .env, source files, path traversal tricks
  3. SQL injection: classic, UNION, stacked, boolean and TIME-BASED payloads in every text field of the public
     endpoints (login, sign-up, email link, resend, two-step) and in admin URL ids. A vulnerable server would
     answer 200/500, leak an SQL error, or sleep; a safe one answers 4xx quickly.
  4. Access control: protected endpoints without a token, with a forged token, with an 'alg: none' token
  5. CSRF defences: missing X-Requested-With header, foreign Origin
  6. Oversized bodies (with and without Content-Length) -> 413
  7. Rate limiting on login -> 429   (uses a made-up username, so no real account is locked)
Each line prints PASS / FAIL / INFO. Exit code 1 if anything FAILs.
Takes about 10-15 minutes against a server with rate limits on (it waits whenever the limit is reached).
The rate-limit test blocks logins from YOUR IP for about a minute. Run it last or with --skip-rate-limit.
"""
import argparse
import base64
import json
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

RESULTS = []
SQL_ERROR_WORDS = ("sqlalchemy", "psycopg", "sqlite", "syntax error", "pg_", "traceback", "operationalerror",
                   "programmingerror", "integrityerror", "unterminated", "sqlstate")


def report(status, name, info=""):
    RESULTS.append(status)
    print(f"  [{status}] {name}" + (f"  ({info})" if info else ""))


def request(base, method, path, body=None, headers=None, raw=None, timeout=20):
    url = base + path
    hdrs = {"User-Agent": "robuststego-security-check"}
    if headers:
        hdrs.update(headers)
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as r:
            return r.status, dict(r.headers), r.read(200_000).decode("utf-8", "replace"), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read(200_000).decode("utf-8", "replace"), time.time() - t0
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        return 0, {}, str(e), time.time() - t0
    except OSError as e:
        # e.g. ConnectionAbortedError / ConnectionResetError: the server answered early (413) and closed the
        # connection while we were still sending. Reported as status 0 = "connection closed by the server".
        return 0, {}, f"connection closed by the server: {type(e).__name__}", time.time() - t0


def api_headers(base):
    return {"X-Requested-With": "fetch", "Origin": base}


def check_headers(base, https):
    print("\n1. Security headers")
    for path, want_csp in (("/", "script-src 'self'"), ("/api/health", "default-src 'none'")):
        st, h, _, _ = request(base, "GET", path)
        h = {k.lower(): v for k, v in h.items()}
        csp = h.get("content-security-policy", "")
        report("PASS" if want_csp in csp and "unsafe-inline" not in csp else "FAIL", f"{path}: Content-Security-Policy", csp[:70])
        report("PASS" if h.get("x-frame-options", "").upper() == "DENY" else "FAIL", f"{path}: X-Frame-Options DENY")
        report("PASS" if h.get("x-content-type-options", "").lower() == "nosniff" else "FAIL", f"{path}: X-Content-Type-Options nosniff")
        report("PASS" if h.get("referrer-policy") == "no-referrer" else "FAIL", f"{path}: Referrer-Policy no-referrer")
        if https:
            report("PASS" if "max-age" in h.get("strict-transport-security", "") else "FAIL", f"{path}: HSTS")
        server = h.get("server", "")
        report("PASS" if "uvicorn" not in server.lower() and "python" not in server.lower() else "FAIL",
               f"{path}: no server software banner", (server or "none") +
               ("  -> start uvicorn with --no-server-header" if "uvicorn" in server.lower() else ""))


def check_closed_doors(base):
    print("\n2. Closed doors (must NOT return the file)")
    for path in ("/docs", "/openapi.json", "/redoc", "/.env", "/.env.example", "/server.py", "/config.py",
                 "/../.env", "/%2e%2e/.env", "/..%2f.env", "/%2e%2e%2f%2e%2e%2fetc/passwd", "/assets/../../.env",
                 "/run_server.bat", "/.git/config", "/stego_auth.db", "/checkpoints_channel_l300/best.pt"):
        st, _, body, _ = request(base, "GET", path)
        leaked = st == 200 and any(k in body for k in ("JWT_SECRET", "DATABASE_URL", "[core]", "import ", "root:", "SQLite format"))
        ok = (st in (400, 401, 403, 404, 405)) or (st == 200 and path in ("/docs", "/redoc") and "swagger" not in body.lower() and "redoc" not in body.lower() and not leaked)
        report("PASS" if ok and not leaked else "FAIL", f"GET {path}", f"HTTP {st}")


SQLI = ["' OR '1'='1", "' OR 1=1 --", "admin'--", "\" OR \"\"=\"", "') OR ('1'='1", "1' UNION SELECT username, password_hash FROM admin_users --",
        "'; DROP TABLE admin_users; --", "' AND 1=CAST((SELECT version()) AS int) --", "%27%20OR%201%3D1--", "ADMIN' /*", "\\' OR 1=1 #"]
SQLI_TIME = ["'; SELECT pg_sleep(6); --", "' OR pg_sleep(6) IS NULL --", "1); SELECT pg_sleep(6); --",
             "' || (SELECT CASE WHEN 1=1 THEN pg_sleep(6) ELSE NULL END) || '", "' AND randomblob(500000000) --"]


def sqli_case(base, name, path, body_fn, ok_codes):
    worst = None
    for p in SQLI + SQLI_TIME:
        st, _, body, dt = request(base, "POST", path, body_fn(p), api_headers(base))
        while st == 429:                 # the rate limit is doing its job; wait for the window and retry this payload
            print(f"    rate limit reached on {path}, waiting 61 s ...", flush=True)
            time.sleep(61)
            st, _, body, dt = request(base, "POST", path, body_fn(p), api_headers(base))
        low = body.lower()
        if st == 200 or st >= 500 or any(w in low for w in SQL_ERROR_WORDS) or dt > 5:
            worst = f"payload {p!r} -> HTTP {st}, {dt:.1f}s, body {body[:80]!r}"
            break
        if st not in ok_codes:
            worst = f"payload {p!r} -> unexpected HTTP {st}"
            break
    report("FAIL" if worst else "PASS", f"{name}: {len(SQLI) + len(SQLI_TIME)} SQL injection payloads", worst or "all refused quickly, no SQL errors")


def check_sqli(base, skip_signup):
    print("\n3. SQL injection (each payload must be refused with 4xx, fast, no database error text)")
    sqli_case(base, "login username", "/api/auth/login", lambda p: {"username": p, "password": "x"}, (400, 401, 422))
    sqli_case(base, "login password", "/api/auth/login", lambda p: {"username": "no_such_user_sec", "password": p}, (400, 401, 422))
    sqli_case(base, "email link token", "/api/auth/verify-email", lambda p: {"token": (p + "x" * 20)[:200]}, (400, 422))
    sqli_case(base, "two-step code", "/api/auth/mfa", lambda p: {"mfa_token": "x" * 20 + p, "code": "123456"}, (400, 401, 422))
    if not skip_signup:
        print("    (sign-up: only 3 tries an hour are allowed, so this samples 3 payloads; expect 400 for each)")
        for p in (SQLI[1], SQLI[5], SQLI_TIME[0]):
            st, _, body, dt = request(base, "POST", "/api/auth/register", {"email": f"a{p}@example.com", "username": p, "password": "Aa1!" + p},
                                      api_headers(base))
            bad = st == 201 or st >= 500 or any(w in body.lower() for w in SQL_ERROR_WORDS) or dt > 5
            report("FAIL" if bad else ("INFO" if st == 429 else "PASS"), f"sign-up payload {p[:30]!r}", f"HTTP {st}, {dt:.1f}s")
    for path in ("/api/admin/users/1%20OR%201=1/approve", "/api/admin/users/1;DROP%20TABLE%20admin_users/disable",
                 "/api/admin/users/-1%20UNION%20SELECT%201/enable"):
        st, _, body, _ = request(base, "POST", path, None, api_headers(base))
        report("PASS" if st in (401, 403, 404, 422) and not any(w in body.lower() for w in SQL_ERROR_WORDS) else "FAIL",
               f"admin URL id {path.split('/')[4][:24]!r}", f"HTTP {st}")


def b64(d):
    return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")


def check_access(base):
    print("\n4. Access control without a valid session")
    forged = b64({"alg": "HS256", "typ": "JWT"}) + "." + b64({"sub": "1", "typ": "access", "exp": 9999999999, "iat": 1}) + ".AAAA"
    alg_none = b64({"alg": "none", "typ": "JWT"}) + "." + b64({"sub": "1", "typ": "access", "exp": 9999999999, "iat": 1}) + "."
    for label, tok in (("no token", None), ("forged signature", forged), ("alg none", alg_none)):
        for method, path in (("GET", "/api/admin/users"), ("POST", "/api/embed"), ("POST", "/api/extract"), ("GET", "/api/auth/me"),
                             ("GET", "/api/auth/mfa/status")):
            h = api_headers(base)
            if tok:
                h["Authorization"] = f"Bearer {tok}"
            st, _, _, _ = request(base, method, path, {} if method == "POST" else None, h)
            report("PASS" if st == 401 else "FAIL", f"{label}: {method} {path}", f"HTTP {st}")


def check_csrf(base):
    print("\n5. CSRF defences on cookie endpoints")
    st, _, _, _ = request(base, "POST", "/api/auth/refresh", None, {"Origin": base})
    report("PASS" if st == 403 else "FAIL", "refresh without X-Requested-With", f"HTTP {st}")
    st, _, _, _ = request(base, "POST", "/api/auth/refresh", None, {"X-Requested-With": "fetch", "Origin": "https://evil.example"})
    report("PASS" if st == 403 else "FAIL", "refresh from a foreign Origin", f"HTTP {st}")


def check_body_limits(base):
    print("\n6. Oversized request bodies")
    big = b'{"image":"' + b"A" * (13 * 1024 * 1024) + b'"}'
    st, _, _, _ = request(base, "POST", "/api/extract", raw=big, headers={**api_headers(base), "Content-Type": "application/json"}, timeout=60)
    report("PASS" if st in (413, 0) else "FAIL", "13 MB body with Content-Length",
           "refused, connection closed early" if st == 0 else f"HTTP {st}")
    # chunked: no Content-Length header at all
    u = urlsplit(base)
    try:
        port = u.port or (443 if u.scheme == "https" else 80)
        sock = socket.create_connection((u.hostname, port), timeout=30)
        if u.scheme == "https":
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=u.hostname)
        head = (f"POST /api/extract HTTP/1.1\r\nHost: {u.netloc}\r\nContent-Type: application/json\r\n"
                f"X-Requested-With: fetch\r\nOrigin: {base}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
        sock.sendall(head.encode())
        chunk = b"A" * (1024 * 1024)
        status = None
        for _ in range(14):
            try:
                sock.sendall(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
            except OSError:
                break
        try:
            sock.sendall(b"0\r\n\r\n")
        except OSError:
            pass
        resp = sock.recv(200).decode("latin-1", "replace")
        status = resp.split(" ")[1] if resp.startswith("HTTP/") else resp[:40]
        sock.close()
        report("PASS" if status in ("413", "400") or status == "" else "FAIL", "14 MB chunked body (no Content-Length)", f"HTTP {status}")
    except OSError as e:
        report("PASS", "14 MB chunked body (no Content-Length)", f"connection closed by the server/proxy: {type(e).__name__}")


def check_rate_limit(base):
    print("\n7. Rate limiting (login; made-up user)")
    codes = []
    for _ in range(8):
        st, _, _, _ = request(base, "POST", "/api/auth/login", {"username": "no_such_user_sec", "password": "Wrong-Pass-123!"}, api_headers(base))
        codes.append(st)
    report("PASS" if 429 in codes else "FAIL", "8 quick wrong logins -> 429 Too Many Requests", str(codes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="e.g. https://laptop-xxxx.tailnet.ts.net or http://localhost:8000")
    ap.add_argument("--skip-rate-limit", action="store_true")
    ap.add_argument("--skip-signup", action="store_true", help="do not use the 3 sign-ups an hour")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    print(f"Security check of {base}  ({time.strftime('%Y-%m-%d %H:%M')})")
    check_headers(base, base.startswith("https://"))
    check_closed_doors(base)
    check_access(base)
    check_csrf(base)
    check_body_limits(base)
    check_sqli(base, a.skip_signup)
    if not a.skip_rate_limit:
        check_rate_limit(base)
    fails = RESULTS.count("FAIL")
    print(f"\nRESULT: {RESULTS.count('PASS')} passed, {fails} failed, {RESULTS.count('INFO')} info")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
