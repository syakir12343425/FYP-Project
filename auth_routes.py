"""/api/auth/*  — login, refresh (rotating, reuse-detecting), logout, me, change-password."""
import uuid
from datetime import timedelta

import jwt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address

import config
from db import AdminUser, RefreshToken, RevokedFamily, User, audit, aware, get_db, utcnow
from security import (DUMMY_HASH, create_access_token, decode_access_token, hash_password,
                      hash_refresh_token, new_refresh_token, password_problem, verify_password)

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/api/auth", tags=["auth"])
bearer = HTTPBearer(auto_error=False)
COOKIE = "refresh_token"


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=1, max_length=128)


def session_body(user: User) -> dict:
    """What login / refresh / change-password return. Role and the must-change flag are for the UI only;
    the server re-checks both from the database on every request."""
    return {"access_token": create_access_token(user.id, user.username), "token_type": "bearer",
            "expires_in": config.ACCESS_TTL_MIN * 60, "username": user.username,
            "role": user.role, "must_change_password": user.must_change_password}


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

    user.failed_attempts, user.locked_until, user.last_login_at = 0, None, now
    db.commit()
    _issue_refresh(db, user, str(uuid.uuid4()), response)
    audit(db, "login_success", request, user.username)
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
    if user is None or user.disabled:
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
    if user is None or user.disabled:          # disabling takes effect at once, not after 15 minutes
        raise HTTPException(401, "Invalid token")
    request.state.username = user.username
    return user


def require_user(user: User = Depends(require_session_user)) -> User:
    """Dependency guarding every protected endpoint."""
    if user.must_change_password:
        raise HTTPException(403, "Password change required")
    return user


def require_admin(user: User = Depends(require_user)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "Admins only")
    return user


@router.get("/me")
def me(user: User = Depends(require_session_user)):
    return {"username": user.username, "role": user.role, "must_change_password": user.must_change_password}


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
