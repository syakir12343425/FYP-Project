"""Run ONCE per database to add multi-user accounts (role, disabled, must_change_password, last_login_at).

    python tools/migrate_users.py

Needs DDL rights: the local SQLite file (done automatically on server start anyway), or the Supabase OWNER
DATABASE_URL in .env. The server's stego_app role cannot do this, by design. Every account that already
exists becomes role 'admin' (they were all made with create_admin.py). Safe to re-run: does nothing then.
After this, re-run supabase_hardening.sql (v3) so stego_app gets the new grants and the admin-protection trigger.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text            # noqa: E402

from db import Base, engine, migrate_users_schema   # noqa: E402

Base.metadata.create_all(engine)
added = migrate_users_schema()
print("Added columns:", ", ".join(added) if added else "none (already up to date)")
with engine.connect() as conn:
    for role, n in conn.execute(text("SELECT role, COUNT(*) FROM admin_users GROUP BY role")):
        print(f"  {role}: {n} account(s)")
