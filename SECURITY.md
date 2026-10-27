# Security design (for the FYP report / viva)

| Threat | Control | Where |
|---|---|---|
| Password theft from DB | bcrypt (cost 12), 12+ char complexity policy | security.py, create_admin.py |
| Brute force / credential stuffing | 5/min/IP rate limit + account lockout (5 fails -> 15 min) | auth_routes.py |
| Username enumeration | Generic error, dummy-hash timing equalisation | auth_routes.py |
| Token theft | 15-min JWT in JS memory only; refresh token httpOnly + SameSite=Strict cookie | app.js, auth_routes.py |
| Stolen/replayed refresh token | Single-use rotation, stored as SHA-256, family revocation on reuse | auth_routes.py |
| JWT forgery / alg=none | Algorithm pinned to HS256, required claims, typ check, 32+ char secret enforced | security.py, config.py |
| CSRF | SameSite=Strict + X-Requested-With header + Origin allow-list | auth_routes.py |
| SQL injection | SQLAlchemy ORM only (parameterised) | db.py |
| Malicious input / DoS | Pydantic Literal/range/length limits, PNG/JPEG only, 32-1024px, decompression-bomb guard, 12 MB body cap, per-endpoint rate limits | server.py |
| XSS / clickjacking / sniffing | CSP, X-Frame-Options, nosniff, no-store, HSTS (when HTTPS) | server.py |
| Data exposure via Supabase REST | RLS on, no policies, anon revoked | supabase_hardening.sql |
| Secret leakage | .env only, startup fails without secrets, never committed | config.py |
| Accountability | Audit log: logins, lockouts, refresh reuse, every embed/downscale/extract | db.py |

## Known limitations (state these honestly)
- Rate-limit counters are in-memory (single process); use Redis if you scale to multiple workers.
- HS256 shared secret; asymmetric (RS256/EdDSA) would be the next step for multi-service setups.
- Single admin, no MFA (TOTP would be a natural future-work item).
- Behind a reverse proxy, configure trusted forwarded headers so the rate limiter sees real client IPs.

## Setup
    pip install -r requirements-auth.txt
    cp .env.example .env      # fill JWT_SECRET (+ DATABASE_URL for Supabase)
    python create_admin.py
    uvicorn server:app --host 0.0.0.0 --port 8000
Serve the frontend on http://localhost:8080 (must match ALLOWED_ORIGINS). For the VPS: put Caddy/Nginx in front for HTTPS, set COOKIE_SECURE=true, ENV=prod, and update API_BASE + ALLOWED_ORIGINS.
