-- =====================================================================================
-- supabase_hardening.sql  (v2)  -  run ONCE in the Supabase SQL editor, as the owner,
-- AFTER `python create_admin.py` has created the tables.
--
-- Before running: replace SCRAM_VERIFIER_HERE with the line printed by
--     python tools/make_db_role_secret.py
-- (a salted one-way hash, so the editor's history never holds the real password).
-- Safe to re-run: every step is idempotent.
-- =====================================================================================
begin;

-- -------------------------------------------------------------------------------------
-- 1. Close the auto-generated REST API (PostgREST) door.
--    Supabase exposes the public schema over HTTP using the public "anon" key.
--    Our app never uses that API, so it gets nothing: RLS on, no policies for anon /
--    authenticated, and all their privileges removed.
-- -------------------------------------------------------------------------------------
alter table public.admin_users      enable row level security;
alter table public.refresh_tokens   enable row level security;
alter table public.revoked_families enable row level security;
alter table public.audit_log        enable row level security;

revoke all on public.admin_users, public.refresh_tokens, public.revoked_families, public.audit_log
    from anon, authenticated;
revoke all on all sequences in schema public from anon, authenticated;

-- -------------------------------------------------------------------------------------
-- 2. Least-privilege application role.
--    The server stops connecting as `postgres` (owner: can drop tables, read everything).
--    `stego_app` can do exactly what the code needs and nothing else:
--      * cannot create / alter / drop tables (no DDL)
--      * cannot create new admin accounts or change password hashes
--      * audit_log is APPEND-ONLY: cannot read, edit or delete log entries
--        -> an attacker who takes over the app cannot erase their tracks.
-- -------------------------------------------------------------------------------------
do $$
begin
    if not exists (select 1 from pg_roles where rolname = 'stego_app') then
        create role stego_app login;
    end if;
end $$;

-- A new role is already NOSUPERUSER, NOCREATEDB, NOCREATEROLE, NOBYPASSRLS by default
-- (Supabase's postgres role may not even be allowed to name some of those attributes).
alter role stego_app with login connection limit 10 password 'SCRAM_VERIFIER_HERE';

grant usage on schema public to stego_app;

-- admin_users: read for login; only the lockout counters may change.
grant select on public.admin_users to stego_app;
grant update (failed_attempts, locked_until) on public.admin_users to stego_app;

-- refresh_tokens: issue new tokens, mark them revoked. No delete, no other column edits.
grant select, insert on public.refresh_tokens to stego_app;
grant update (revoked) on public.refresh_tokens to stego_app;
grant usage on sequence public.refresh_tokens_id_seq to stego_app;

-- revoked_families: append-only kill list.
grant select, insert on public.revoked_families to stego_app;

-- audit_log: insert only. SELECT on the id column alone is needed because the ORM does
-- INSERT ... RETURNING id; it cannot read events, usernames, IPs or details.
grant insert on public.audit_log to stego_app;
grant select (id) on public.audit_log to stego_app;
grant usage on sequence public.audit_log_id_seq to stego_app;

-- -------------------------------------------------------------------------------------
-- 3. RLS policies for stego_app.
--    stego_app does not own the tables and has no BYPASSRLS, so RLS applies to it.
--    These policies let its rows through; the GRANTs above still decide which
--    operations and columns are allowed.
-- -------------------------------------------------------------------------------------
drop policy if exists stego_app_all on public.admin_users;
create policy stego_app_all on public.admin_users      for all to stego_app using (true) with check (true);

drop policy if exists stego_app_all on public.refresh_tokens;
create policy stego_app_all on public.refresh_tokens   for all to stego_app using (true) with check (true);

drop policy if exists stego_app_all on public.revoked_families;
create policy stego_app_all on public.revoked_families for all to stego_app using (true) with check (true);

drop policy if exists stego_app_all on public.audit_log;
create policy stego_app_all on public.audit_log        for all to stego_app using (true) with check (true);

commit;

-- -------------------------------------------------------------------------------------
-- 4. Check (read-only). Expected: rls_enabled = true for all 4 tables, and stego_app
--    privileges exactly as granted above.
-- -------------------------------------------------------------------------------------
select c.relname as table_name, c.relrowsecurity as rls_enabled
from pg_class c join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public'
  and c.relname in ('admin_users', 'refresh_tokens', 'revoked_families', 'audit_log')
order by 1;
