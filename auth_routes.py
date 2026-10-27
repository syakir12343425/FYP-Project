"""/api/auth/*  — login, refresh (rotating, reuse-detecting), logout, me."""
import uuid
from datetime import timedelta

import jwt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address

import config
from db import AdminUser, RefreshToken, audit, aware, get_db, utcnow
from security import (DUMMY_HASH, create_access_token, decode_access_token, hash_refresh_token,
                      new_refresh_token, verify_password)

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/api/auth", tags=["auth"])
bearer = HTTPBearer(auto_error=False)
COOKIE = "refresh_token"


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


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

    user.failed_attempts, user.locked_until = 0, None
    db.commit()
    _issue_refresh(db, user, str(uuid.uuid4()), response)
    audit(db, "login_success", request, user.username)
    return {"access_token": create_access_token(user.id, user.username), "token_type": "bearer",
            "expires_in": config.ACCESS_TTL_MIN * 60, "username": user.username}


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
    if rec.revoked:
        # An already-rotated token was replayed => likely theft. Kill the whole chain.
        db.execute(update(RefreshToken).where(RefreshToken.family_id == rec.family_id).values(revoked=True))
        db.commit()
        audit(db, "refresh_reuse_detected", request, detail=f"family={rec.family_id}")
        response.delete_cookie(COOKIE, path="/api/auth")
        raise HTTPException(401, "Session revoked")
    if aware(rec.expires_at) < utcnow():
        raise HTTPException(401, "Session expired")
    user = db.get(AdminUser, rec.user_id)
    if user is None:
        raise HTTPException(401, "Invalid session")
    rec.revoked = True                      # rotate: old token is single-use
    db.commit()
    _issue_refresh(db, user, rec.family_id, response)
    audit(db, "token_refreshed", request, user.username)
    return {"access_token": create_access_token(user.id, user.username), "token_type": "bearer",
            "expires_in": config.ACCESS_TTL_MIN * 60, "username": user.username}


@router.post("/logout")
def logout(request: Request, response: Response, refresh_token: str | None = Cookie(default=None),
           db: Session = Depends(get_db)):
    _csrf_guard(request)
    if refresh_token:
        rec = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(refresh_token)))
        if rec:
            db.execute(update(RefreshToken).where(RefreshToken.family_id == rec.family_id).values(revoked=True))
            db.commit()
            audit(db, "logout", request)
    response.delete_cookie(COOKIE, path="/api/auth")
    return {"ok": True}


def require_user(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(bearer),
                 db: Session = Depends(get_db)) -> AdminUser:
    """Dependency guarding every protected endpoint."""
    if creds is None:
        raise HTTPException(401, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    try:
        data = decode_access_token(creds.credentials)
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    user = db.get(AdminUser, int(data["sub"]))
    if user is None:
        raise HTTPException(401, "Invalid token")
    request.state.username = user.username
    return user


@router.get("/me")
def me(user: AdminUser = Depends(require_user)):
    return {"username": user.username}
