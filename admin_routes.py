"""/api/admin/*  — account management for role 'admin'. Normal users get 403 on every route.

Design choices (for the report):
  * No public sign-up: only an admin creates accounts, with a random one-time password that the server
    generates and shows ONCE. The user must choose their own password at first login.
  * Admin accounts cannot be created, reset or disabled here; that needs create_admin.py with database
    owner rights. On Supabase a trigger enforces this even if this server is compromised.
  * Accounts are disabled, never deleted, so the audit log keeps pointing at a real account.
  * Temporary passwords are never written to the audit log.
"""
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth_routes import kill_user_sessions, limiter, require_admin, retire_mfa
from db import User, audit, aware, get_db, utcnow
from security import hash_password, temporary_password

router = APIRouter(prefix="/api/admin", tags=["admin"])
USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,31}$")


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)


def _public(u: User) -> dict:
    locked = bool(u.locked_until and aware(u.locked_until) > utcnow())
    return {"id": u.id, "username": u.username, "role": u.role, "disabled": u.disabled, "locked": locked,
            "must_change_password": u.must_change_password, "mfa_enabled": u.mfa_enabled,
            "created_at": aware(u.created_at).isoformat() if u.created_at else None,
            "last_login_at": aware(u.last_login_at).isoformat() if u.last_login_at else None}


def _normal_user(db: Session, user_id: int) -> User:
    u = db.get(User, user_id)
    if u is None:
        raise HTTPException(404, "No such account")
    if u.role == "admin":
        raise HTTPException(403, "Admin accounts are managed with create_admin.py")
    return u


@router.get("/users")
@limiter.limit("60/minute")
def list_users(request: Request, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.scalars(select(User).order_by(User.role, User.username)).all()
    return {"users": [_public(u) for u in users]}


@router.post("/users", status_code=201)
@limiter.limit("20/minute")
def create_user(request: Request, body: CreateUserRequest, admin: User = Depends(require_admin),
                db: Session = Depends(get_db)):
    name = body.username.strip()
    if not USERNAME_RE.match(name):
        raise HTTPException(400, "Username: 3 to 32 characters, letters, digits, dot, dash or underscore, "
                                 "starting with a letter or digit")
    # Case-insensitive duplicate check (not ILIKE: "_" and "%" would act as wildcards there).
    if db.scalar(select(User.id).where(func.lower(User.username) == name.lower())) is not None:
        raise HTTPException(409, "That username is taken")
    temp = temporary_password()
    u = User(username=name, password_hash=hash_password(temp), role="user", must_change_password=True)
    db.add(u)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "That username is taken")
    audit(db, "user_created", request, admin.username, f"new_user={name}")
    return {"user": _public(u), "temporary_password": temp}


@router.post("/users/{user_id}/reset-password")
@limiter.limit("20/minute")
def reset_password(request: Request, user_id: int, admin: User = Depends(require_admin),
                   db: Session = Depends(get_db)):
    u = _normal_user(db, user_id)
    temp = temporary_password()
    u.password_hash, u.must_change_password = hash_password(temp), True
    u.failed_attempts, u.locked_until = 0, None
    db.commit()
    kill_user_sessions(db, u.id, "admin_reset")
    audit(db, "user_password_reset", request, admin.username, f"user={u.username}")
    return {"user": _public(u), "temporary_password": temp}


@router.post("/users/{user_id}/disable")
@limiter.limit("30/minute")
def disable_user(request: Request, user_id: int, admin: User = Depends(require_admin),
                 db: Session = Depends(get_db)):
    u = _normal_user(db, user_id)
    u.disabled = True
    db.commit()
    kill_user_sessions(db, u.id, "disabled")
    audit(db, "user_disabled", request, admin.username, f"user={u.username}")
    return {"user": _public(u)}


@router.post("/users/{user_id}/enable")
@limiter.limit("30/minute")
def enable_user(request: Request, user_id: int, admin: User = Depends(require_admin),
                db: Session = Depends(get_db)):
    u = _normal_user(db, user_id)
    u.disabled = False
    db.commit()
    audit(db, "user_enabled", request, admin.username, f"user={u.username}")
    return {"user": _public(u)}


@router.post("/users/{user_id}/unlock")
@limiter.limit("30/minute")
def unlock_user(request: Request, user_id: int, admin: User = Depends(require_admin),
                db: Session = Depends(get_db)):
    u = db.get(User, user_id)          # admins may be unlocked too: only the lockout counters change
    if u is None:
        raise HTTPException(404, "No such account")
    u.failed_attempts, u.locked_until = 0, None
    db.commit()
    audit(db, "user_unlocked", request, admin.username, f"user={u.username}")
    return {"user": _public(u)}


@router.post("/users/{user_id}/reset-mfa")
@limiter.limit("20/minute")
def reset_mfa(request: Request, user_id: int, admin: User = Depends(require_admin),
              db: Session = Depends(get_db)):
    """Lost phone: turn two-step verification off so the person can sign in with their password and set it
    up again. Signs them out everywhere. Admins are reset with create_admin.py instead."""
    u = _normal_user(db, user_id)
    retire_mfa(db, u)
    db.commit()
    kill_user_sessions(db, u.id, "mfa_reset")
    audit(db, "user_mfa_reset", request, admin.username, f"user={u.username}")
    return {"user": _public(u)}
