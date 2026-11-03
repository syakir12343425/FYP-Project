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


AdminUser = User   # old name, kept so existing imports keep working


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
    return missing


def init_db():
    Base.metadata.create_all(engine)
    if missing_user_columns():
        if engine.dialect.name == "sqlite":
            migrate_users_schema()          # local dev database: just do it
        else:
            # The server's least-privilege role (stego_app) has no DDL rights, by design.
            raise RuntimeError("Database is missing the multi-user columns. Run once with the OWNER "
                               "DATABASE_URL:  python tools/migrate_users.py  (see SECURITY.md)")


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
