"""/api/auth/*  — login (+ two-step code), refresh (rotating, reuse-detecting), logout, me, change-password,
two-step verification set-up / turn off / new recovery codes."""
import hashlib
import re
import secrets
import uuid
from datetime import timedelta

import jwt
from fastapi import APIRouter, BackgroundTasks, Cookie, Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address

import config
from mailer import already_registered_email, mail_configured, send_mail_quietly, verification_email
from db import AdminUser, MfaRecoveryCode, RefreshToken, RevokedFamily, User, audit, aware, get_db, utcnow
from security import (DUMMY_HASH, create_access_token, create_mfa_token, decode_access_token, decode_mfa_token,
                      decrypt_mfa_secret, encrypt_mfa_secret, hash_password, hash_recovery_code,
                      hash_refresh_token, new_recovery_codes, new_refresh_token, normalise_recovery_code,
                      otpauth_uri, password_problem, qr_data_uri, totp_new_secret, totp_verify, verify_password)

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/api/auth", tags=["auth"])
bearer = HTTPBearer(auto_error=False)
COOKIE = "refresh_token"


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=128)


class TokenRequest(BaseModel):
    token: str = Field(min_length=20, max_length=200)


class MfaLoginRequest(BaseModel):
    mfa_token: str = Field(min_length=10, max_length=2048)
    code: str = Field(min_length=6, max_length=32)


class PasswordConfirm(BaseModel):
    password: str = Field(min_length=1, max_length=128)


class CodeRequest(BaseModel):
    code: str = Field(min_length=6, max_length=32)


class PasswordAndCode(BaseModel):
    password: str = Field(min_length=1, max_length=128)
    code: str = Field(min_length=6, max_length=32)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=1, max_length=128)


def session_body(user: User) -> dict:
    """What login / refresh / change-password return. Role and the must-change flag are for the UI only;
    the server re-checks both from the database on every request."""
    return {"access_token": create_access_token(user.id, user.username), "token_type": "bearer",
            "expires_in": config.ACCESS_TTL_MIN * 60, "username": user.username,
            "role": user.role, "must_change_password": user.must_change_password,
            "mfa_enabled": user.mfa_enabled, "mfa_mandatory": mfa_required_for(user)}


def mfa_required_for(user: User) -> bool:
    return config.REQUIRE_MFA_ALL or (user.role == "admin" and config.REQUIRE_ADMIN_MFA)


def mask_email(email: str | None) -> str:
    if not email or "@" not in email:
        return "your email"
    local, domain = email.split("@", 1)
    return f"{local[0]}***@{domain}"


def _csrf_guard(request: Request):
    """Cookie-authenticated endpoint => defend against CSRF: custom header (forces CORS preflight)
    plus Origin allow-list, on top of SameSite=Strict."""
    if request.headers.get("x-requested-with") != "fetch":
        raise HTTPException(403, "Missing CSRF header")
    origin = request.headers.get("origin")
    if origin and origin not in config.ALLOWED_ORIGINS:
        raise HTTPException(403, "Origin not allowed")


def _set_cookie(resp: Response, raw: str):
    resp.set_cookie(COOKIE, raw, max_age=config.REFRESH_TTL_DAYS * 86400, httponly=True,
                    secure=config.COOKIE_SECURE, samesite="strict", path="/api/auth")


def _issue_refresh(db: Session, user: AdminUser, family: str, resp: Response):
    raw, h = new_refresh_token()
    db.add(RefreshToken(user_id=user.id, token_hash=h, family_id=family,
                        expires_at=utcnow() + timedelta(days=config.REFRESH_TTL_DAYS)))
    db.commit()
    _set_cookie(resp, raw)


def _kill_family(db: Session, family_id: str, reason: str):
    """Revoke every token in a family AND record the family as dead, so tokens inserted into it
    later (by a request racing with this one) are rejected too."""
    for _ in range(2):
        db.execute(update(RefreshToken).where(RefreshToken.family_id == family_id).values(revoked=True))
        if db.get(RevokedFamily, family_id) is None:
            db.add(RevokedFamily(family_id=family_id, reason=reason))
        try:
            db.commit()
            return
        except IntegrityError:      # another request recorded the same family first - that's fine
            db.rollback()


def kill_user_sessions(db: Session, user_id: int, reason: str):
    """Log a user out everywhere: kill every refresh-token family they have. Their current 15-minute
    access token stops working too, because require_user re-reads the account on every request."""
    families = db.scalars(select(RefreshToken.family_id).where(RefreshToken.user_id == user_id).distinct()).all()
    for fam in families:
        _kill_family(db, fam, reason)


@router.post("/login")
@limiter.limit("5/minute")
def login(request: Request, body: LoginRequest, response: Response, db: Session = Depends(get_db)):
    _csrf_guard(request)
    user = db.scalar(select(AdminUser).where(AdminUser.username == body.username))
    now = utcnow()

    if user and user.locked_until and aware(user.locked_until) > now:
        verify_password(body.password, DUMMY_HASH)   # keep timing uniform
        audit(db, "login_blocked_locked", request, body.username)
        raise HTTPException(401, "Invalid credentials or account temporarily locked")

    ok = verify_password(body.password, user.password_hash if user else DUMMY_HASH) and user is not None
    if not ok:
        if user:
            user.failed_attempts += 1
            if user.failed_attempts >= config.MAX_FAILED_LOGINS:
                user.locked_until = now + timedelta(minutes=config.LOCKOUT_MINUTES)
                user.failed_attempts = 0
                audit(db, "account_locked", request, user.username)
            db.commit()
        audit(db, "login_failed", request, body.username[:64])
        raise HTTPException(401, "Invalid credentials or account temporarily locked")   # generic message

    if user.disabled:
        # Checked only AFTER a correct password, with the same generic answer, so it reveals nothing
        # to someone guessing passwords.
        audit(db, "login_blocked_disabled", request, user.username)
        raise HTTPException(401, "Invalid credentials or account temporarily locked")

    # The next two answers are given only to someone who already has the right password, so saying them
    # plainly is safe and helpful.
    if not user.email_verified:
        audit(db, "login_blocked_unverified", request, user.username)
        raise HTTPException(403, f"Confirm your email first: open the link we sent to {mask_email(user.email)}")
    if user.pending_approval:
        audit(db, "login_blocked_pending", request, user.username)
        raise HTTPException(403, "Your account is waiting for an admin to approve it")

    if user.mfa_enabled:
        # Password right, second step still to come. No session yet; the failure counter is NOT reset,
        # so wrong codes keep counting towards the lockout.
        audit(db, "login_password_ok_mfa_pending", request, user.username)
        # "mfa_required" = enter your code now. (The site-wide rule is a different field: "mfa_mandatory".)
        return {"mfa_required": True, "mfa_token": create_mfa_token(user.id)}

    user.failed_attempts, user.locked_until, user.last_login_at = 0, None, now
    db.commit()
    _issue_refresh(db, user, str(uuid.uuid4()), response)
    audit(db, "login_success", request, user.username)
    return session_body(user)


def _register_failure(db: Session, request: Request, user: User, event: str):
    now = utcnow()
    user.failed_attempts += 1
    if user.failed_attempts >= config.MAX_FAILED_LOGINS:
        user.locked_until = now + timedelta(minutes=config.LOCKOUT_MINUTES)
        user.failed_attempts = 0
        audit(db, "account_locked", request, user.username)
    db.commit()
    audit(db, event, request, user.username)


def check_second_factor(db: Session, user: User, code: str) -> str | None:
    """A current authenticator code or an unused recovery code. Returns 'totp' / 'recovery_code', or None.
    Both are claimed ATOMICALLY in the database (UPDATE ... WHERE still unused), so two requests racing
    with the same code cannot both succeed."""
    if not user.mfa_enabled or not user.mfa_secret_enc:
        return None
    step = totp_verify(decrypt_mfa_secret(user.mfa_secret_enc, user.id), code, user.mfa_last_step)
    if step is not None:
        claimed = db.execute(update(User).where(User.id == user.id, User.mfa_last_step < step)
                             .values(mfa_last_step=step)).rowcount
        db.commit()
        if claimed == 1:
            user.mfa_last_step = step
            return "totp"
        return None
    norm = normalise_recovery_code(code)
    if norm:
        rec = db.scalar(select(MfaRecoveryCode).where(MfaRecoveryCode.user_id == user.id,
                                                      MfaRecoveryCode.code_hash == hash_recovery_code(norm),
                                                      MfaRecoveryCode.used_at.is_(None)))
        if rec is not None:
            claimed = db.execute(update(MfaRecoveryCode).where(MfaRecoveryCode.id == rec.id,
                                                               MfaRecoveryCode.used_at.is_(None))
                                 .values(used_at=utcnow())).rowcount
            db.commit()
            if claimed == 1:
                return "recovery_code"
    return None


def _replace_recovery_codes(db: Session, user: User) -> list[str]:
    """Retire every unused code (marked used, never deleted) and store 10 new ones as keyed hashes."""
    db.execute(update(MfaRecoveryCode).where(MfaRecoveryCode.user_id == user.id, MfaRecoveryCode.used_at.is_(None))
               .values(used_at=utcnow()))
    codes = new_recovery_codes()
    for c in codes:
        db.add(MfaRecoveryCode(user_id=user.id, code_hash=hash_recovery_code(c)))
    return codes


def retire_mfa(db: Session, user: User):
    """Turn two-step verification off for this user (their choice, or an admin reset after a lost phone)."""
    user.mfa_enabled, user.mfa_secret_enc, user.mfa_pending_enc, user.mfa_pending_at = False, None, None, None
    db.execute(update(MfaRecoveryCode).where(MfaRecoveryCode.user_id == user.id, MfaRecoveryCode.used_at.is_(None))
               .values(used_at=utcnow()))


@router.post("/mfa")
@limiter.limit("5/minute")
def login_mfa(request: Request, body: MfaLoginRequest, response: Response, db: Session = Depends(get_db)):
    """Second step of signing in: the code from the authenticator app (or a recovery code)."""
    _csrf_guard(request)
    try:
        data = decode_mfa_token(body.mfa_token)
    except jwt.PyJWTError:
        raise HTTPException(401, "Sign-in took too long. Start again.")
    user = db.get(User, int(data["sub"]))
    now = utcnow()
    if user is None or user.disabled or user.pending_approval or not user.mfa_enabled:
        raise HTTPException(401, "Invalid credentials or account temporarily locked")
    if user.locked_until and aware(user.locked_until) > now:
        audit(db, "login_blocked_locked", request, user.username)
        raise HTTPException(401, "Invalid credentials or account temporarily locked")
    how = check_second_factor(db, user, body.code)
    if how is None:
        _register_failure(db, request, user, "mfa_failed")
        raise HTTPException(401, "Wrong code")
    user.failed_attempts, user.locked_until, user.last_login_at = 0, None, now
    db.commit()
    _issue_refresh(db, user, str(uuid.uuid4()), response)
    audit(db, "login_success", request, user.username, f"second_factor={how}")
    return session_body(user)


@router.post("/refresh")
@limiter.limit("30/minute")
def refresh(request: Request, response: Response, refresh_token: str | None = Cookie(default=None),
            db: Session = Depends(get_db)):
    _csrf_guard(request)
    if not refresh_token:
        raise HTTPException(401, "No session")
    rec = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(refresh_token)))
    if rec is None:
        raise HTTPException(401, "Invalid session")
    if db.get(RevokedFamily, rec.family_id) is not None:
        # Family already killed (reuse detected or logged out) - even a token issued after the kill.
        raise HTTPException(401, "Session revoked")
    if rec.revoked:
        # An already-rotated token was replayed => likely theft. Kill the whole chain.
        _kill_family(db, rec.family_id, "reuse")
        audit(db, "refresh_reuse_detected", request, detail=f"family={rec.family_id}")
        response.delete_cookie(COOKIE, path="/api/auth")
        raise HTTPException(401, "Session revoked")
    if aware(rec.expires_at) < utcnow():
        raise HTTPException(401, "Session expired")
    user = db.get(User, rec.user_id)
    if user is None or user.disabled or user.pending_approval or not user.email_verified:
        raise HTTPException(401, "Invalid session")
    # Rotate: the old token is single-use. Claim it ATOMICALLY in the database: only the request
    # whose UPDATE flips revoked False->True wins. A plain read-then-write ("if not rec.revoked:
    # rec.revoked = True") lets two concurrent requests with a stolen token both pass the check.
    claimed = db.execute(
        update(RefreshToken)
        .where(RefreshToken.id == rec.id, RefreshToken.revoked.is_(False))
        .values(revoked=True)
    ).rowcount
    db.commit()
    if claimed != 1:
        # Another request used this token first => same as replay: kill the whole chain.
        _kill_family(db, rec.family_id, "reuse")
        audit(db, "refresh_reuse_detected", request, detail=f"family={rec.family_id} (concurrent)")
        response.delete_cookie(COOKIE, path="/api/auth")
        raise HTTPException(401, "Session revoked")
    _issue_refresh(db, user, rec.family_id, response)
    audit(db, "token_refreshed", request, user.username)
    return session_body(user)


@router.post("/logout")
def logout(request: Request, response: Response, refresh_token: str | None = Cookie(default=None),
           db: Session = Depends(get_db)):
    _csrf_guard(request)
    if refresh_token:
        rec = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(refresh_token)))
        if rec:
            _kill_family(db, rec.family_id, "logout")
            audit(db, "logout", request)
    response.delete_cookie(COOKIE, path="/api/auth")
    return {"ok": True}


def require_session_user(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(bearer),
                         db: Session = Depends(get_db)) -> User:
    """Valid token for an enabled account. Allows accounts that still must change their password, so only
    use it for /me and /change-password."""
    if creds is None:
        raise HTTPException(401, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    try:
        data = decode_access_token(creds.credentials)
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    user = db.get(User, int(data["sub"]))
    if user is None or user.disabled or user.pending_approval or not user.email_verified:   # at once, not after 15 min
        raise HTTPException(401, "Invalid token")
    request.state.username = user.username
    return user


def require_setup_user(user: User = Depends(require_session_user)) -> User:
    """Signed in and own password chosen. Allowed: setting up two-step verification (and its status)."""
    if user.must_change_password:
        raise HTTPException(403, "Password change required")
    return user


def require_user(user: User = Depends(require_setup_user)) -> User:
    """Dependency guarding every protected endpoint: also needs two-step verification where it is required."""
    if mfa_required_for(user) and not user.mfa_enabled:
        raise HTTPException(403, "Set up two-step verification (Account page) to continue")
    return user


def require_admin(user: User = Depends(require_user)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "Admins only")
    return user


@router.get("/me")
def me(user: User = Depends(require_session_user)):
    return {"username": user.username, "role": user.role, "must_change_password": user.must_change_password,
            "mfa_enabled": user.mfa_enabled, "mfa_mandatory": mfa_required_for(user)}


@router.post("/change-password")
@limiter.limit("5/minute")
def change_password(request: Request, body: ChangePasswordRequest, response: Response,
                    user: User = Depends(require_session_user), db: Session = Depends(get_db)):
    _csrf_guard(request)
    if user.role == "admin":
        # The database refuses this for the app's role anyway (trigger in supabase_hardening.sql).
        raise HTTPException(403, "Admin passwords are changed with create_admin.py")
    if not verify_password(body.current_password, user.password_hash):
        audit(db, "password_change_failed", request, user.username)
        raise HTTPException(400, "Current password is wrong")
    problem = password_problem(body.new_password, user.username)
    if problem:
        raise HTTPException(400, problem)
    if verify_password(body.new_password, user.password_hash):
        raise HTTPException(400, "Choose a password different from the current one")
    user.password_hash, user.must_change_password = hash_password(body.new_password), False
    db.commit()
    kill_user_sessions(db, user.id, "password_change")     # every other device is logged out
    _issue_refresh(db, user, str(uuid.uuid4()), response)  # this device continues with a fresh session
    audit(db, "password_changed", request, user.username)
    return session_body(user)


# ------------------------------------------------------------------ two-step verification management
@router.post("/mfa/setup")
@limiter.limit("5/minute")
def mfa_setup(request: Request, body: PasswordConfirm, user: User = Depends(require_setup_user),
              db: Session = Depends(get_db)):
    """Start set-up: needs the password again (a stolen, unlocked session alone cannot enrol a device).
    The new secret is kept as 'pending' until a correct code proves the app was set up."""
    _csrf_guard(request)
    if user.mfa_enabled:
        raise HTTPException(409, "Two-step verification is already on")
    if not verify_password(body.password, user.password_hash):
        audit(db, "mfa_setup_bad_password", request, user.username)
        raise HTTPException(400, "Password is wrong")
    secret = totp_new_secret()
    user.mfa_pending_enc, user.mfa_pending_at = encrypt_mfa_secret(secret, user.id), utcnow()
    db.commit()
    audit(db, "mfa_setup_started", request, user.username)
    uri = otpauth_uri(secret, user.username)
    return {"secret": secret, "otpauth_uri": uri, "qr_data_uri": qr_data_uri(uri)}


@router.post("/mfa/enable")
@limiter.limit("5/minute")
def mfa_enable(request: Request, body: CodeRequest, user: User = Depends(require_setup_user),
               db: Session = Depends(get_db)):
    _csrf_guard(request)
    if user.mfa_enabled:
        raise HTTPException(409, "Two-step verification is already on")
    if not user.mfa_pending_enc or not user.mfa_pending_at or aware(user.mfa_pending_at) < utcnow() - timedelta(minutes=10):
        raise HTTPException(400, "Set-up expired. Start again.")
    step = totp_verify(decrypt_mfa_secret(user.mfa_pending_enc, user.id), body.code, 0)
    if step is None:
        audit(db, "mfa_enable_bad_code", request, user.username)
        raise HTTPException(400, "Wrong code. Check that your phone's clock is set automatically, then try the newest code.")
    user.mfa_secret_enc, user.mfa_pending_enc, user.mfa_pending_at = user.mfa_pending_enc, None, None
    user.mfa_enabled, user.mfa_last_step = True, step
    codes = _replace_recovery_codes(db, user)
    db.commit()
    audit(db, "mfa_enabled", request, user.username)
    return {"mfa_enabled": True, "recovery_codes": codes}


@router.post("/mfa/disable")
@limiter.limit("5/minute")
def mfa_disable(request: Request, body: PasswordAndCode, user: User = Depends(require_user),
                db: Session = Depends(get_db)):
    _csrf_guard(request)
    if not user.mfa_enabled:
        raise HTTPException(409, "Two-step verification is already off")
    if mfa_required_for(user):
        raise HTTPException(403, "Two-step verification is required on this site and cannot be turned off")
    if not verify_password(body.password, user.password_hash) or check_second_factor(db, user, body.code) is None:
        audit(db, "mfa_disable_failed", request, user.username)
        raise HTTPException(400, "Password or code is wrong")
    retire_mfa(db, user)
    db.commit()
    audit(db, "mfa_disabled", request, user.username)
    return {"mfa_enabled": False}


@router.post("/mfa/recovery-codes")
@limiter.limit("5/minute")
def mfa_new_recovery_codes(request: Request, body: PasswordAndCode, user: User = Depends(require_user),
                           db: Session = Depends(get_db)):
    _csrf_guard(request)
    if not user.mfa_enabled:
        raise HTTPException(409, "Two-step verification is off")
    if not verify_password(body.password, user.password_hash) or check_second_factor(db, user, body.code) is None:
        audit(db, "mfa_recovery_codes_failed", request, user.username)
        raise HTTPException(400, "Password or code is wrong")
    codes = _replace_recovery_codes(db, user)
    db.commit()
    audit(db, "mfa_recovery_codes_replaced", request, user.username)
    return {"recovery_codes": codes}


@router.get("/mfa/status")
def mfa_status(user: User = Depends(require_setup_user), db: Session = Depends(get_db)):
    left = db.scalar(select(func.count()).select_from(MfaRecoveryCode)
                     .where(MfaRecoveryCode.user_id == user.id, MfaRecoveryCode.used_at.is_(None)))
    return {"mfa_enabled": user.mfa_enabled, "recovery_codes_left": int(left or 0),
            "required": mfa_required_for(user)}


# ------------------------------------------------------------------ self sign-up
USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,31}$")


EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$")


def _new_email_token(user: User) -> str:
    raw = secrets.token_urlsafe(32)
    user.email_token_hash = hashlib.sha256(raw.encode()).hexdigest()
    user.email_token_expires = utcnow() + timedelta(hours=config.EMAIL_TOKEN_HOURS)
    return raw


def _send_verification(background: BackgroundTasks, user: User, raw: str):
    # The token rides in the URL #fragment: browsers never send that to a server or in a Referer header.
    subject, text = verification_email(user.username, f"{config.PUBLIC_URL}/#verify={raw}")
    background.add_task(send_mail_quietly, user.email, subject, text)


@router.post("/register", status_code=201)
@limiter.limit("3/hour")
def register(request: Request, body: RegisterRequest, background: BackgroundTasks, db: Session = Depends(get_db)):
    """Anyone may create an account with an email address. It works only after the link sent to that address is
    opened, and the app is usable only after two-step verification is set up at first sign-in.
    Against bots and abuse: 3 sign-ups per hour per IP, and sign-ups pause while MAX_PENDING_SIGNUPS accounts
    are still unconfirmed (or unapproved). An email address that already has an account gets the same answer
    as a new one (and its owner gets a heads-up email), so the form does not reveal who is registered."""
    _csrf_guard(request)
    if not config.ALLOW_SIGNUP:
        raise HTTPException(403, "Sign-up is closed. Ask an admin for an account.")
    if not mail_configured() and config.ENV != "dev":
        raise HTTPException(503, "Sign-up is not available right now.")
    name, email = body.username.strip(), body.email.strip().lower()
    if not USERNAME_RE.match(name):
        raise HTTPException(400, "Username: 3 to 32 characters, letters, digits, dot, dash or underscore, "
                                 "starting with a letter or digit")
    if not EMAIL_RE.match(email):
        raise HTTPException(400, "Enter a valid email address")
    problem = password_problem(body.password, name)
    if problem:
        raise HTTPException(400, problem)
    waiting = db.scalar(select(func.count()).select_from(User).where(
        (User.pending_approval.is_(True)) | ((User.email_verified.is_(False)) & (User.email.is_not(None)))))
    if (waiting or 0) >= config.MAX_PENDING_SIGNUPS:
        audit(db, "signup_paused_queue_full", request, name)
        raise HTTPException(503, "Sign-ups are paused for now. Try again later.")
    if db.scalar(select(User.id).where(func.lower(User.username) == name.lower())) is not None:
        hash_password(body.password)        # similar timing to a real sign-up
        raise HTTPException(409, "That username is taken")
    done = {"ok": True, "message": "Check your email for a link to confirm your account."}
    existing = db.scalar(select(User).where(User.email == email))
    if (existing is not None and not existing.email_verified and existing.email_token_expires
            and aware(existing.email_token_expires) < utcnow()):
        # Someone registered this address and never confirmed it (maybe not the owner). Once their link has
        # expired, the address is released so its real owner can sign up; that stale account stays disabled.
        existing.email, existing.disabled, existing.email_token_hash = None, True, None
        db.commit()
        audit(db, "signup_released_stale_email", request, existing.username)
        existing = None
    if existing is not None:
        hash_password(body.password)
        subject, text = already_registered_email(existing.username)
        background.add_task(send_mail_quietly, email, subject, text)
        audit(db, "signup_existing_email", request, name)
        return done
    u = User(username=name, email=email, email_verified=False, password_hash=hash_password(body.password),
             role="user", pending_approval=config.SIGNUP_REQUIRES_APPROVAL)
    raw = _new_email_token(u)
    db.add(u)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "That username is taken")
    _send_verification(background, u, raw)
    audit(db, "signup_requested", request, name)
    return done


@router.post("/verify-email")
@limiter.limit("10/minute")
def verify_email(request: Request, body: TokenRequest, db: Session = Depends(get_db)):
    _csrf_guard(request)
    h = hashlib.sha256(body.token.encode()).hexdigest()
    u = db.scalar(select(User).where(User.email_token_hash == h))
    if u is None or not u.email_token_expires or aware(u.email_token_expires) < utcnow():
        raise HTTPException(400, "This link is invalid or has expired. Sign in to get a new one.")
    claimed = db.execute(update(User).where(User.id == u.id, User.email_token_hash == h)     # single use, atomic
                         .values(email_verified=True, email_token_hash=None, email_token_expires=None)).rowcount
    db.commit()
    if claimed != 1:
        raise HTTPException(400, "This link is invalid or has expired. Sign in to get a new one.")
    audit(db, "email_verified", request, u.username)
    return {"ok": True, "message": "Email confirmed. You can sign in now."}


@router.post("/resend-verification")
@limiter.limit("3/hour")
def resend_verification(request: Request, body: LoginRequest, background: BackgroundTasks,
                        db: Session = Depends(get_db)):
    """New confirmation link. Needs the password, so nobody can make the app email other people's addresses."""
    _csrf_guard(request)
    u = db.scalar(select(User).where(User.username == body.username))
    ok = verify_password(body.password, u.password_hash if u else DUMMY_HASH)
    if not ok or u is None or u.disabled:
        raise HTTPException(401, "Invalid credentials or account temporarily locked")
    if u.email_verified or not u.email:
        raise HTTPException(409, "This account is already confirmed")
    raw = _new_email_token(u)
    db.commit()
    _send_verification(background, u, raw)
    audit(db, "email_verification_resent", request, u.username)
    return {"ok": True, "message": f"A new link was sent to {mask_email(u.email)}."}


@router.get("/signup-open")
def signup_open():
    return {"open": config.ALLOW_SIGNUP and (mail_configured() or config.ENV == "dev")}
