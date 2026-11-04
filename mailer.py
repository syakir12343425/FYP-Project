"""Outgoing email (sign-up confirmation). Gmail SMTP with an app password, set in .env:

    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=yourname@gmail.com
    SMTP_PASSWORD=<16-letter Google app password>      (never your real Gmail password)
    MAIL_FROM=RobustStego <yourname@gmail.com>
    PUBLIC_URL=http://localhost:8080                  (where the UI is opened; used to build links)

Without SMTP settings, and only when ENV=dev, the email is printed to the server console instead, so sign-up
can be tried locally. In production without SMTP settings, sending fails loudly (no silent fake success).
"""
import smtplib
import ssl
from email.message import EmailMessage

import config


def mail_configured() -> bool:
    return bool(config.SMTP_HOST and config.SMTP_USER and config.SMTP_PASSWORD and config.MAIL_FROM)


def send_mail(to: str, subject: str, body: str) -> None:
    if not mail_configured():
        if config.ENV == "dev":
            print(f"\n--- [dev mail, not sent] to: {to}\nSubject: {subject}\n\n{body}\n--- end ---\n", flush=True)
            return
        raise RuntimeError("Email is not configured (SMTP_* settings in .env)")
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = config.MAIL_FROM, to, subject
    msg.set_content(body)
    ctx = ssl.create_default_context()                      # verifies the server certificate
    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=15) as s:
        s.starttls(context=ctx)                             # never send the password in clear
        s.login(config.SMTP_USER, config.SMTP_PASSWORD)
        s.send_message(msg)


def send_mail_quietly(to: str, subject: str, body: str) -> None:
    """For background tasks: a failed send is logged on the server, never shown to the person (so the sign-up
    answer cannot reveal anything about the address)."""
    try:
        send_mail(to, subject, body)
    except Exception as e:                                   # noqa: BLE001 - log and carry on
        print(f"[mail] sending to {to!r} failed: {type(e).__name__}: {e}", flush=True)


def verification_email(username: str, link: str) -> tuple[str, str]:
    return ("Confirm your RobustStego account",
            f"Hello {username},\n\n"
            f"Open this link to confirm your email address and finish creating your account:\n\n{link}\n\n"
            f"The link works once and expires in {config.EMAIL_TOKEN_HOURS} hours.\n"
            "After you confirm, sign in and set up two-step verification with an authenticator app.\n\n"
            "If you did not sign up, ignore this email: nothing happens without the link.\n")


def already_registered_email(username: str) -> tuple[str, str]:
    return ("Someone tried to sign up with your email",
            f"Hello {username},\n\n"
            "Someone tried to create a new RobustStego account with this email address, which already has an "
            "account. If it was you, just sign in instead. If not, you can ignore this email.\n")
