# Security design (for the FYP report / viva)

| Threat | Control | Where |
|---|---|---|
| Password theft from DB | bcrypt (cost 12), 12+ char complexity policy | security.py, create_admin.py |
| Brute force / credential stuffing | 5/min/IP rate limit + account lockout (5 fails -> 15 min) | auth_routes.py |
| Username enumeration | Generic error, dummy-hash timing equalisation | auth_routes.py |
| Token theft | 15-min JWT in JS memory only; refresh token httpOnly + SameSite=Strict cookie | app.js, auth_routes.py |
| Stolen/replayed refresh token | Single-use rotation, stored as SHA-256, family revocation on reuse | auth_routes.py |
| Concurrent replay race (TOCTOU) | Atomic single-use claim (`UPDATE ... WHERE revoked = false`); killed families recorded in `revoked_families` so tokens inserted later are also rejected; frontend single-flight refresh | auth_routes.py, db.py, app.js - verified by tests/test_refresh_race.py (50/50 rounds) |
| JWT forgery / alg=none | Algorithm pinned to HS256, required claims, typ check, 32+ char secret enforced | security.py, config.py |
| CSRF | SameSite=Strict + X-Requested-With header + Origin allow-list | auth_routes.py |
| SQL injection | SQLAlchemy ORM only (parameterised) | db.py |
| Malicious input / DoS | Pydantic Literal/range/length limits, PNG/JPEG only, 32-1280px, decompression-bomb guard, 12 MB body cap, per-endpoint rate limits | server.py |
| XSS / clickjacking / sniffing | CSP, X-Frame-Options, nosniff, no-store, HSTS (when HTTPS) | server.py |
| Data exposure via Supabase REST | RLS on all 4 tables, no policies for anon/authenticated, all their privileges revoked | supabase_hardening.sql |
| Compromised app server abusing the DB | App connects as least-privilege role `stego_app` (not the owner): no DDL, cannot create admins or change password hashes, no deletes | supabase_hardening.sql - verified by tests/test_db_privileges.py (17/17) |
| Attacker erasing their tracks | audit_log is append-only for `stego_app`: INSERT only, cannot read, update or delete entries | supabase_hardening.sql |
| DB password leaking via Supabase SQL history | Password generated locally; only a SCRAM-SHA-256 verifier (salted hash) is sent to Supabase | tools/make_db_role_secret.py |
| Database traffic interception | TLS enforced (`sslmode=require`) | .env |
| Secret leakage | .env only, startup fails without secrets, never committed | config.py |
| Accountability | Audit log: logins, lockouts, refresh reuse, every embed/downscale/extract | db.py |

## Known limitations (state these honestly)
- Rate-limit counters are in-memory (single process); use Redis if you scale to multiple workers.
- HS256 shared secret; asymmetric (RS256/EdDSA) would be the next step for multi-service setups.
- Single admin, no MFA (TOTP would be a natural future-work item).
- Behind a reverse proxy, configure trusted forwarded headers so the rate limiter sees real client IPs.
- In a refresh race, the winning request still receives one 15-minute access token before reuse is detected (JWTs are stateless and cannot be recalled).
- tests/test_refresh_race.py runs on a throwaway SQLite database; true concurrency on Postgres is not separately tested.
- Expired refresh tokens and old audit entries are not purged yet (a scheduled cleanup job and a retention period are future work).
- Supabase Free plan pauses after ~1 week of inactivity: open the dashboard before any demo.

## Setup

### Local development (SQLite, no internet needed)
    pip install -r requirements-auth.txt
    copy .env.example .env    # fill JWT_SECRET; keep the sqlite DATABASE_URL
    python create_admin.py
    uvicorn server:app --port 8000

### Supabase (hosted Postgres, least privilege)
1. Create the project (region: Southeast Asia / Singapore). Connect > Direct > **Session pooler** (port 5432).
2. In `.env`, set DATABASE_URL to the **owner** URL (`postgres.<ref>`, `postgresql+psycopg2://`, `?sslmode=require`).
3. `python create_admin.py` - creates the tables and the admin account (needs owner rights).
4. `python tools/make_db_role_secret.py` - run ONCE; prints a SCRAM verifier and a stego_app DATABASE_URL.
5. Supabase SQL editor: paste supabase_hardening.sql, replace SCRAM_VERIFIER_HERE with the verifier, Run.
   Expected: rls_enabled = true for all 4 tables.
6. In `.env`, comment out the owner line and make the stego_app line (same run as step 4) the active one.
7. `python tests/test_db_privileges.py` - expect 17/17.

Owner credentials are only for setup/admin tasks (creating or resetting the admin, schema changes); the running server never uses them.

### Production (VPS)
Serve the frontend on the origin listed in ALLOWED_ORIGINS. Put Caddy/Nginx in front for HTTPS, set COOKIE_SECURE=true, ENV=prod, and update API_BASE in app.js + ALLOWED_ORIGINS.
