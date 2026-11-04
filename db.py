"""SQLAlchemy models. All queries go through the ORM (parameterised) -> no SQL injection surface."""
from datetime import datetime, timezone
from sqlalchemy import create_engine, false, inspect, text, String, Integer, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
import config


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def aware(dt: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; Postgres returns aware ones. Normalise to UTC-aware."""
    return dt if dt is None or dt.tzinfo else dt.replace(tzinfo=timezone.utc)


_kw = {"pool_pre_ping": True}
if config.DATABASE_URL.startswith("sqlite"):
    _kw["connect_args"] = {"check_same_thread": False}
engine = create_engine(config.DATABASE_URL, **_kw)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


ROLES = ("admin", "user")


class User(Base):
    """Every account. The table keeps its old name (admin_users) so the Supabase grants, RLS policies and
    triggers in supabase_hardening.sql keep pointing at the same table.
    role "admin": created only with create_admin.py (database owner rights); manages accounts in the app.
    role "user":  created by an admin in the app; can use Hide / Read / Results only."""
    __tablename__ = "admin_users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))          # bcrypt, never plaintext
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    role: Mapped[str] = mapped_column(String(16), default="user", server_default="user")
    disabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    # Set when an admin creates the account or resets its password: the temporary password works only
    # for choosing a new one (every other endpoint answers 403 until then).
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Two-step verification (TOTP). Secrets are AES-GCM encrypted with MFA_ENCRYPTION_KEY (security.py).
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    mfa_secret_enc: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mfa_pending_enc: Mapped[str | None] = mapped_column(String(255), nullable=True)   # during set-up only
    mfa_pending_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    mfa_last_step: Mapped[int] = mapped_column(Integer, default=0, server_default="0")  # replay protection
    # Self sign-up: True until an admin approves the account. A pending account cannot sign in.
    pending_approval: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    # Email (self sign-up). Stored lower-case; unique. Accounts made before this, or by an admin, count as
    # verified. The confirmation token is stored only as a SHA-256 hash, single-use, with an expiry.
    email: Mapped[str | None] = mapped_column(String(254), nullable=True, unique=True, index=True)
    # ORM default True: only the self sign-up path creates an unconfirmed account (it sets False explicitly).
    email_verified: Mapped[bool] = mapped_column(Boolean, default=True, server_default=false())
    email_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    email_token_expires: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


AdminUser = User


class MfaRecoveryCode(Base):
    """One-time codes for a lost phone. Stored as keyed hashes; used codes are marked, never deleted."""
    __tablename__ = "mfa_recovery_codes"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("admin_users.id", ondelete="CASCADE"), index=True)
    code_hash: Mapped[str] = mapped_column(String(64), index=True)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)   # old name, kept so existing imports keep working


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("admin_users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)   # SHA-256 of the token
    family_id: Mapped[str] = mapped_column(String(36), index=True)                 # rotation chain (reuse detection)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RevokedFamily(Base):
    """A refresh-token family that has been killed (reuse detected, or logout).
    Revoking only the existing token rows is not enough: a concurrent request can still INSERT a
    new token into the family a moment later. Checking this table on every refresh closes that gap."""
    __tablename__ = "revoked_families"
    family_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    reason: Mapped[str] = mapped_column(String(32))
    revoked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    event: Mapped[str] = mapped_column(String(48), index=True)
    username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(200), nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)


# Columns added for multi-user accounts (Nov 2026). Existing databases get them via migrate_users_schema().
_NEW_USER_COLUMNS = {
    "role":                 ("VARCHAR(16) NOT NULL DEFAULT 'user'", "VARCHAR(16) NOT NULL DEFAULT 'user'"),
    "disabled":             ("BOOLEAN NOT NULL DEFAULT 0", "BOOLEAN NOT NULL DEFAULT false"),
    "must_change_password": ("BOOLEAN NOT NULL DEFAULT 0", "BOOLEAN NOT NULL DEFAULT false"),
    "last_login_at":        ("DATETIME", "TIMESTAMP WITH TIME ZONE"),
    # two-step verification (Nov 2026)
    "mfa_enabled":          ("BOOLEAN NOT NULL DEFAULT 0", "BOOLEAN NOT NULL DEFAULT false"),
    "mfa_secret_enc":       ("VARCHAR(255)", "VARCHAR(255)"),
    "mfa_pending_enc":      ("VARCHAR(255)", "VARCHAR(255)"),
    "mfa_pending_at":       ("DATETIME", "TIMESTAMP WITH TIME ZONE"),
    "mfa_last_step":        ("INTEGER NOT NULL DEFAULT 0", "INTEGER NOT NULL DEFAULT 0"),
    # self sign-up (Nov 2026)
    "pending_approval":     ("BOOLEAN NOT NULL DEFAULT 0", "BOOLEAN NOT NULL DEFAULT false"),
    "email":                ("VARCHAR(254)", "VARCHAR(254)"),
    "email_verified":       ("BOOLEAN NOT NULL DEFAULT 0", "BOOLEAN NOT NULL DEFAULT false"),
    "email_token_hash":     ("VARCHAR(64)", "VARCHAR(64)"),
    "email_token_expires":  ("DATETIME", "TIMESTAMP WITH TIME ZONE"),
}


def missing_user_columns(eng=engine) -> list[str]:
    insp = inspect(eng)
    if not insp.has_table("admin_users"):
        return []
    have = {c["name"] for c in insp.get_columns("admin_users")}
    return [c for c in _NEW_USER_COLUMNS if c not in have]


def migrate_users_schema(eng=engine) -> list[str]:
    """Add the multi-user columns to an existing admin_users table. Needs DDL rights (SQLite file, or the
    Supabase OWNER url). Every account that existed before this migration was made by create_admin.py,
    so it becomes role 'admin'. Returns the columns that were added."""
    missing = missing_user_columns(eng)
    sqlite = eng.dialect.name == "sqlite"
    with eng.begin() as conn:
        for col in missing:
            ddl = _NEW_USER_COLUMNS[col][0 if sqlite else 1]
            conn.execute(text(f"ALTER TABLE admin_users ADD COLUMN {col} {ddl}"))
        if "role" in missing:
            conn.execute(text("UPDATE admin_users SET role = 'admin'"))
        if "email_verified" in missing:      # accounts that existed before email sign-up need no confirmation
            conn.execute(text("UPDATE admin_users SET email_verified = " + ("1" if sqlite else "true")))
        if "email" in missing:
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_admin_users_email_unique ON admin_users (email)"))
    return missing


def init_db():
    sqlite = engine.dialect.name == "sqlite"
    if not sqlite:
        # Check BEFORE create_all: the server's least-privilege role (stego_app) has no DDL rights, by design,
        # so a missing table or column must be added by the owner (SQL editor), not here.
        insp = inspect(engine)
        if insp.has_table("admin_users") and (missing_user_columns() or not insp.has_table("mfa_recovery_codes")):
            raise RuntimeError("Database is missing the account columns (multi-user / two-step verification). "
                               "Run the newest supabase_*_v*.sql upgrade in the Supabase SQL editor (see SECURITY.md)")
    Base.metadata.create_all(engine)
    if sqlite and missing_user_columns():
        migrate_users_schema()          # local dev database: just do it


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def audit(db, event: str, request=None, username: str | None = None, detail: str | None = None):
    ip = ua = None
    if request is not None:
        ip = request.client.host if request.client else None
        ua = (request.headers.get("user-agent") or "")[:200]
    db.add(AuditLog(event=event, username=username, ip=ip, user_agent=ua, detail=(detail or "")[:1000] or None))
    db.commit()
