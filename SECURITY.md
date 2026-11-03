# Security design (for the FYP report / viva)

| Threat | Control | Where |
|---|---|---|
| Password theft from DB | bcrypt (cost 12), 12+ char complexity policy (no username inside), one policy function for every path | security.py |
| Unwanted accounts | No public sign-up: only an admin creates accounts; server-generated 16-char one-time password shown once, never logged | admin_routes.py |
| Temporary password left in use | `must_change_password`: every app endpoint answers 403 until the user picks their own password | auth_routes.py (require_user) |
| Normal user doing admin things | Role checked from the database on every request (`require_admin`), not from the token or the UI | auth_routes.py, admin_routes.py |
| Leaver / compromised account | Admin disables the account: refresh families killed, and the live 15-min access token stops at once (account re-read per request) | admin_routes.py, auth_routes.py |
| Forgotten password | Admin reset: new one-time password, lockout cleared, all sessions killed | admin_routes.py |
| Password change hijack | Needs the current password; logs out every other device | auth_routes.py |
| Compromised app server creating or taking over admins | stego_app may INSERT only role 'user'; a trigger blocks changing an admin's password, disabled flag or must-change flag; role and username are not updatable at all | supabase_hardening.sql v3 - tests/test_db_privileges.py |
| Brute force / credential stuffing | 5/min/IP rate limit + account lockout (5 fails -> 15 min) | auth_routes.py |
| Username enumeration | Generic error, dummy-hash timing equalisation | auth_routes.py |
| Token theft | 15-min JWT in JS memory only; refresh token httpOnly + SameSite=Strict cookie | app.js, auth_routes.py |
| Stolen/replayed refresh token | Single-use rotation, stored as SHA-256, family revocation on reuse | auth_routes.py |
| Concurrent replay race (TOCTOU) | Atomic single-use claim (`UPDATE ... WHERE revoked = false`); killed families recorded in `revoked_families` so tokens inserted later are also rejected; frontend single-flight refresh | auth_routes.py, db.py, app.js - verified by tests/test_refresh_race.py (50/50 rounds) |
| JWT forgery / alg=none | Algorithm pinned to HS256, required claims, typ check, 32+ char secret enforced | security.py, config.py |
| CSRF | SameSite=Strict + X-Requested-With header + Origin allow-list | auth_routes.py |
| SQL injection | SQLAlchemy ORM only (parameterised) | db.py |
| Malicious input / DoS | Pydantic Literal/range/length limits, PNG/JPEG/WebP only (animated files rejected), 32-1280px, decompression-bomb guard, 12 MB body cap, per-endpoint rate limits | server.py |
| XSS / clickjacking / sniffing | CSP, X-Frame-Options, nosniff, no-store, HSTS (when HTTPS) | server.py |
| Data exposure via Supabase REST | RLS on all 4 tables, no policies for anon/authenticated, all their privileges revoked | supabase_hardening.sql |
| Compromised app server abusing the DB | App connects as least-privilege role `stego_app` (not the owner): no DDL, no deletes, can manage normal users only (see above) | supabase_hardening.sql - verified by tests/test_db_privileges.py (23 checks) |
| Attacker erasing their tracks | audit_log is append-only for `stego_app`: INSERT only, cannot read, update or delete entries | supabase_hardening.sql |
| DB password leaking via Supabase SQL history | Password generated locally; only a SCRAM-SHA-256 verifier (salted hash) is sent to Supabase | tools/make_db_role_secret.py |
| Database traffic interception | TLS enforced (`sslmode=require`) | .env |
| Secret leakage | .env only, startup fails without secrets, never committed | config.py |
| Accountability | Audit log: logins, lockouts, refresh reuse, every embed/downscale/extract, every account action (who did what to whom) | db.py |

## Known limitations (state these honestly)
- Rate-limit counters are in-memory (single process); use Redis if you scale to multiple workers.
- HS256 shared secret; asymmetric (RS256/EdDSA) would be the next step for multi-service setups.
- No MFA (TOTP would be a natural future-work item).
- Accounts are never deleted (disable instead), so the table grows; fine at FYP scale.
- The temporary password reaches the user through the admin (in person or a message), outside the app.
- The audit log is append-only for the app, so it cannot be shown in the app; read it in the Supabase dashboard (owner).
- tests/test_users.py runs on SQLite; the Postgres trigger is checked by tests/test_db_privileges.py.
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
    python tests/test_users.py      # accounts, roles, forced password change (throwaway DB)

### Accounts
- Admins: `python create_admin.py` only (owner rights on Supabase).
- Everyone else: sign in as admin > Users > Create account. Give the person the one-time password;
  they choose their own at first sign-in. Reset / Disable / Unlock are on the same page.

### Upgrading an existing database to multi-user (once)
- SQLite: nothing to do, the server adds the columns on start (existing accounts become admins).
- Supabase: paste supabase_multiuser_v3.sql into the SQL editor and Run (it adds the columns, the grants
  and the trigger, and does not touch the stego_app password), then `python tests/test_db_privileges.py`.
  Checked on a local Postgres 16 with the v2 setup: re-runnable; stego_app can create/reset/disable normal
  users only; creating, resetting, disabling or promoting to admin is refused.

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
