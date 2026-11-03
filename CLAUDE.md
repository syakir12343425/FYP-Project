# RobustStego: FYP project guide for Claude Code

Final year project (B.Sc. Information Security, USIM) by Muhammad Syakir ("Kirrr"), supervised by
Dr. Syaril Nizam Omar. Thesis: **"Robust Deep Image Steganography Against Downscaling Operations in AI
Pipelines"**. Supervisor deadline: **Sun 22 Nov 2026**.

A short text message (max 63 bytes) is hidden in a 512×512 photo so it can still be read after the photo
is resized (AI pipelines, e.g. 512 → 224 px) or sent through WhatsApp. Four methods are compared: a trained
CNN encoder/decoder (the contribution) and three classical baselines (LSB, DCT, DWT).

## Rules for working in this repo (the owner's standing instructions)

1. **Shell commands are PowerShell** (Windows laptop). Use `;` not `&&`, `$env:VAR`, backslash paths.
2. **Explain step by step**; the owner is a student. Be honest and direct; push back on bad ideas.
3. **Never overclaim.** Every number shown anywhere (UI, report, slides) must trace to a real measurement
   (a tool in `tools/`, `eval.py`, or live inference). Never invent, round up, or "estimate" a result.
   The measured constants in `fyp-stego-app/app.js` (`REFERENCE`, `BENCHMARK_BY_SIZE`, `STICKER_RESULTS`,
   `REAL_SEND`) are copied from the Results log; do not change them in a design task.
4. **Secrets:** never open, print or ask for `.env`, `*.db`, passwords, `JWT_SECRET`, `MFA_ENCRYPTION_KEY`,
   database URLs, TOTP setup keys or recovery codes. If one appears in the conversation, tell the owner to
   rotate it. Never commit `.env` or `*.db` (they are in `.gitignore`; check `git status` before committing).
5. **Small, targeted edits** over rewriting whole files. These files are large (app.js ~2,100 lines,
   index.html ~1,250, styles.css ~1,500): read the part you change, change only that.
6. **Test before handing over** (see Testing). Say plainly what was and was not tested.
7. If a change would weaken security or make results misleading, say so and propose the honest alternative.
8. Commit/push only when the owner asks.

## Layout

| Path | What it is |
|---|---|
| `fyp-stego-app/` | The web UI: **vanilla HTML/CSS/JS, no framework, no build step** (`index.html`, `app.js`, `styles.css`, `assets/samples/*.jpg`). Served by `python -m http.server 8080`. |
| `server.py` | FastAPI inference server on :8000: `/api/embed`, `/api/downscale`, `/api/extract`. Loads the CNN **model D** = `checkpoints_channel_l300/best.pt`. |
| `auth_routes.py`, `admin_routes.py`, `security.py`, `db.py`, `config.py` | Accounts: login, refresh tokens, roles (admin/user), two-step verification (TOTP), admin user management. |
| `models.py`, `downscale.py`, `losses.py`, `train.py`, `dataset.py`, `eval.py`, `configs/*.yaml` | CNN training (done on Colab). |
| `classical_stego.py`, `lsb_baseline.py`, `payload.py`, `rs.py` | Baselines and the payload (Reed-Solomon RS(128,64) + whitening, 1024 bits). |
| `tools/` | Experiments: `benchmark_methods.py`, `sticker_test.py`, `steganalysis.py`, ...; `dev_mock_api.py` = UI test server (below). |
| `tests/` | `test_users.py`, `test_mfa.py` (SQLite, no torch), `test_db_privileges.py` (Supabase), `test_payload.py`, `test_classical.py`, `test_refresh_race.py`. |
| `supabase_*.sql`, `SECURITY.md` | Database hardening (least-privilege role `stego_app`, RLS, triggers) and the security design. |

## Running it

```powershell
# real server (needs torch, the checkpoint and a filled .env)
uvicorn server:app --port 8000
# UI
cd fyp-stego-app; python -m http.server 8080      # then open http://localhost:8080
```
After changing `app.js` or `styles.css`, the browser may keep the old copy: **Ctrl + Shift + R**.

### Working on the UI without the model or a database: `tools/dev_mock_api.py`
A test-only stand-in for server.py on 127.0.0.1:8000. It runs the **real** LSB, DCT and DWT code (CNN requests
fall back to LSB), signs you in automatically and fakes the account and two-step endpoints. Use it for design
work; never present its output as a result.
```powershell
python tools\dev_mock_api.py                       # admin, two-step already on
$env:MOCK_ROLE="user"; python tools\dev_mock_api.py   # normal user
$env:MOCK_LOGGEDOUT="1"; $env:MOCK_MFA="1"; python tools\dev_mock_api.py   # shows the login + code step
```
Mock sign-in password `Right-Pass-123!`, mock code `123456`. Stop the real server first (same port).

## Frontend architecture (read before editing the UI)

- **One page, several views**: each `<section class="viewport" id="page-…">` is a page; `handleNavigation(id)`
  shows one. Pages: `page-home` (Start: Hide/Read tabs + drop zone), `page-embedding` → `page-result` →
  `page-downscaling` → `page-extraction` (the 4 Hide steps), `page-read`, `page-results`, `page-users`
  (admins), `page-account` (two-step, password).
- `body[data-mode]` = `home | hide | read | results | users | account` (from `pageMode()`); CSS uses it to
  show the Hide stepper only in `hide` mode. `body[data-role]` = `admin | user`; classes `.admin-only` /
  `.user-only` hide elements. **The UI only hides; the server enforces every permission.**
- Any `button[data-goto="page-x"]` navigates (bound once in `initApp()`, so it must exist in the HTML).
- State lives in `state` (Hide flow) and `readState` (Read flow). Auth: the access token is kept **only in a JS
  variable** (`accessToken`), never in localStorage; the refresh token is an httpOnly cookie.
  Use `apiFetch()` for protected GET/POST, `sessionPost()` for endpoints that also need the CSRF header.
- **Security rule for UI code:** text from the server or from a decoded image (recovered messages,
  usernames) is untrusted. Insert it with `textContent`, never `innerHTML`. Images from the server go into
  `<img src>` only.
- Errors: `showInlineError()` / `clearInlineError()` next to the control; `errorDetail(res)` reads FastAPI's
  `{"detail": ...}`.

## Design system (current state)

- Dark and light themes: colour tokens on `:root` / `body.light-theme` in `styles.css` (`--bg`, `--surface`,
  `--surface-2/3`, `--text`, `--text-2/3`, `--border`, `--accent` #2f6fe8, `--success`, `--warning`, `--danger`,
  `--*-soft`). **Method colours** (used in charts and labels; keep them): `--cnn`, `--lsb`, `--dct`, `--dwt`.
- Spacing `--space-1…7`, radii `--radius-sm/md/lg/full`, type `--text-xs…xl`, fonts `--font-sans` (Inter),
  `--font-mono` (JetBrains Mono).
- Layout (Nov 2026 redesign, "online tool" style like TinyPNG): sticky top bar (logo, Hide / Read / Results /
  Users / Account, theme switch, Log out), centred content max 1200 px, Start page = tabs + one big drop zone,
  Read page = large image left + result right, site footer with the thesis credit.
- Accessibility already passes: contrast ≥ 4.5:1, focus outlines, skip link, labels, 24 px targets, focus
  moves to the page heading on navigation, no horizontal scroll at 390 px. Keep it that way.
- Writing style in the UI: sentence case, plain words ("Hide a message", not "Embed payload"), say what
  happened and what to do next, no ALL-CAPS labels.

### Known design to-dos (owner wants a design pass)
- Cache-busting: add `?v=<date>` to the `styles.css` / `app.js` links in `index.html`.
- Remaining ALL-CAPS / uppercase labels (`.nav-heading`, `.academic-meta .meta-label`, table headers) → sentence case.
- Typography: one deliberate family (the earlier proposal was Atkinson Hyperlegible Next + Mono); currently Inter
  loaded from Google Fonts (blocked offline → system fallback).
- The Hide flow pages (steps 1–4) still use the older card-heavy layout; Results page could use the same
  calmer style as Start/Read.
- Keep: the measured numbers, the method colours, the security behaviour, the textContent rule.

## Testing

- Backend, no torch needed: `python tests\test_users.py` (expects all pass) and `python tests\test_mfa.py`.
- Supabase privileges (with the stego_app URL active): `python tests\test_db_privileges.py`.
- UI: run `tools\dev_mock_api.py` + `python -m http.server 8080`, click through Start → Hide (all 4 steps) →
  Read → Results → Users → Account in **both themes** and at a narrow width (~390 px). Check the browser console
  for errors. If Playwright is available, take screenshots at 1440×900.
- After any UI change, re-check: no console errors, focus visible, nothing overflows at 390 px, text contrast.

## Background facts (for wording, not for new claims)

- Model D on DIV2K: mean BER 1.76 %, PSNR 35.54 dB, SSIM 0.9767, exact recovery 88.9 %. DCT 4.19 % / 40.12 dB;
  DWT 7.14 % / 40.76 dB; LSB 45.43 % (fails under any resize).
- WhatsApp stickers are WebP quality 90. Real sends: CNN (model A) exact at 2.73 % BER, DCT exact at 0.68 %,
  LSB failed (52.15 %).
- Steganalysis (Results log 9e): DCT/DWT/CNN are detectable by a trained detector (AUC 0.93–0.98); the CNN hides
  in colour; there is **no secret key**, so anyone with the code can read a message (known limitation).
- The full measurement history is in the owner's "Results log" document, not in this repo.
