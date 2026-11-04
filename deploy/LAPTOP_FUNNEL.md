# Running RobustStego publicly from the laptop (Tailscale Funnel)

```
Visitor --HTTPS--> Tailscale Funnel --(encrypted tunnel, started from the laptop)--> 127.0.0.1:8000 on the laptop
                                                                                     uvicorn: web app + /api
```
Online only while the laptop is on, awake, and the server window is open.

## One-time setup
1. Install Tailscale for Windows (tailscale.com/download/windows), sign in (free personal plan).
2. In the Tailscale admin console: DNS > enable **MagicDNS** and **HTTPS certificates**; Access controls > allow
   **Funnel** for this machine (the console offers a button the first time `tailscale funnel` is run).
3. Find the address: `tailscale status` / `tailscale funnel status` -> `https://<laptop>.<tailnet>.ts.net`.
4. `.env` (edit by hand; never share it):
   ```
   ENV=prod
   COOKIE_SECURE=true
   PUBLIC_URL=https://<laptop>.<tailnet>.ts.net
   ALLOWED_ORIGINS=https://<laptop>.<tailnet>.ts.net,http://localhost:8000
   SERVE_UI=true
   ```
   plus the SMTP_* lines (Gmail app password) and the keys you already have.

## Every time
```powershell
cd C:\Users\syakir\FYP\fyp_stego
.\tools\start_public.ps1          # checks .env, opens the public address, starts the server
.\tools\start_public.ps1 -Stop    # afterwards: close the public address
```
If PowerShell refuses to run scripts: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (once).

## Security notes (for the report)
- The laptop opens no ports and the router is untouched; the server listens on 127.0.0.1 only.
- HTTPS certificate and TLS are handled by Tailscale; the laptop's home IP is not revealed to visitors.
- The visitor's real IP reaches the app's rate limits and audit log through X-Forwarded-For, trusted only from
  127.0.0.1 (uvicorn --forwarded-allow-ips 127.0.0.1).
- Risk: the app runs on a personal computer. Keep Windows updated, stop the funnel when not demoing, and do not
  keep unrelated sensitive files in the project folder. A VPS (deploy/Caddyfile + robuststego.service) isolates this.
