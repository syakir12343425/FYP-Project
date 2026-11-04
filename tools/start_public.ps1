# Start RobustStego for people on the internet, from this laptop, through Tailscale Funnel.
#   .\tools\start_public.ps1          start (server + public HTTPS address)
#   .\tools\start_public.ps1 -Stop    take the public address down again
# The server listens on 127.0.0.1 only; Tailscale Funnel is the one way in from outside.
param([switch]$Stop)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$ts = (Get-Command tailscale -ErrorAction SilentlyContinue).Source
if (-not $ts) { $ts = "C:\Program Files\Tailscale\tailscale.exe" }
if (-not (Test-Path $ts)) { Write-Host "Tailscale is not installed: https://tailscale.com/download/windows" -ForegroundColor Red; exit 1 }

if ($Stop) {
    & $ts funnel reset
    Write-Host "Public address is OFF. (Stop the server window with Ctrl+C.)" -ForegroundColor Yellow
    exit 0
}

# Check the production settings in .env without printing any secret values.
if (-not (Test-Path .env)) { Write-Host ".env is missing" -ForegroundColor Red; exit 1 }
$cfg = @{}
foreach ($line in Get-Content .env) {
    if ($line -match '^\s*([A-Z_]+)\s*=\s*(.*)\s*$') { $cfg[$Matches[1]] = $Matches[2] }
}
$problems = @()
if ($cfg["ENV"] -ne "prod") { $problems += "ENV=prod (closes /docs and the dev email fallback)" }
if ($cfg["COOKIE_SECURE"] -ne "true") { $problems += "COOKIE_SECURE=true (cookie only over HTTPS)" }
if (-not ($cfg["PUBLIC_URL"] -like "https://*.ts.net")) { $problems += "PUBLIC_URL=https://<your-laptop>.<tailnet>.ts.net" }
if (-not ($cfg["ALLOWED_ORIGINS"] -like "*https://*.ts.net*")) { $problems += "ALLOWED_ORIGINS must include the same https://...ts.net address" }
if (-not $cfg["SMTP_PASSWORD"]) { $problems += "SMTP_* settings (sign-up emails cannot be sent without them)" }
if ($problems.Count -gt 0) {
    Write-Host "Fix these lines in .env first:" -ForegroundColor Red
    $problems | ForEach-Object { Write-Host "  - $_" }
    exit 1
}

& $ts funnel --bg 8000
Write-Host "Public address: $($cfg['PUBLIC_URL'])   (stop it with .\tools\start_public.ps1 -Stop)" -ForegroundColor Green
Write-Host "Starting the server. Keep this window open; Ctrl+C stops it." -ForegroundColor Green
# --proxy-headers + trusting only 127.0.0.1: the visitor's real IP (from Funnel) reaches the rate limits and audit log.
uvicorn server:app --host 127.0.0.1 --port 8000 --proxy-headers --forwarded-allow-ips 127.0.0.1 --no-server-header
