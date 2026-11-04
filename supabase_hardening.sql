-- =====================================================================================
-- supabase_hardening.sql  (v5: multi-user accounts + two-step verification + email sign-up, Nov 2026)  -  run ONCE in the Supabase SQL editor, as the owner,
-- AFTER `python create_admin.py` has created the tables
-- (upgrading from v2: run `python tools/migrate_users.py` with the owner URL first).
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
--      * can create NORMAL user accounts and set their passwords (admin page in the app),
--        but can never create, promote, reset or disable an ADMIN account (trigger in step 3b)
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

-- admin_users: read for login; lockout counters and last login change on every login.
-- v3: an admin creates normal users in the app, users choose their own password, an admin can
-- disable them. Username, role and created_at can never be updated; no deletes.
grant select on public.admin_users to stego_app;
revoke update on public.admin_users from stego_app;
grant update (failed_attempts, locked_until, last_login_at,
              password_hash, must_change_password, disabled) on public.admin_users to stego_app;
grant insert (username, password_hash, role, disabled, must_change_password,
              failed_attempts, locked_until, created_at, last_login_at) on public.admin_users to stego_app;
grant usage on sequence public.admin_users_id_seq to stego_app;

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

-- -------------------------------------------------------------------------------------
-- 3b. Admin accounts are out of the app's reach (v3).
--     Column grants cannot say "only for rows where role = 'user'", so a trigger does it.
--     For stego_app only (the owner, used by create_admin.py, is not affected):
--       * INSERT must be role 'user'                       -> no new admins
--       * an admin row's password_hash / disabled /
--         must_change_password cannot change               -> no taking over or locking out the admin
--     (role and username are not in the UPDATE grant at all -> no promotion, no renaming)
--     So even with the server fully compromised, an attacker gets at most normal accounts.
-- -------------------------------------------------------------------------------------
create or replace function public.protect_admin_accounts() returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
begin
    if current_user <> 'stego_app' then
        return new;
    end if;
    if tg_op = 'INSERT' then
        if new.role is distinct from 'user' then
            raise exception 'stego_app may only create normal user accounts' using errcode = '42501';
        end if;
        return new;
    end if;
    if new.role is distinct from old.role then
        raise exception 'stego_app may not change roles' using errcode = '42501';
    end if;
    if old.role = 'admin' and (new.password_hash is distinct from old.password_hash
                               or new.disabled is distinct from old.disabled
                               or new.must_change_password is distinct from old.must_change_password
                               or new.email is distinct from old.email
                               or new.email_verified is distinct from old.email_verified) then
        raise exception 'stego_app may not change admin accounts' using errcode = '42501';
    end if;
    if old.role = 'admin' and old.mfa_enabled and (new.mfa_enabled is distinct from old.mfa_enabled
                                                   or new.mfa_secret_enc is distinct from old.mfa_secret_enc) then
        raise exception 'stego_app may not turn off or replace an admin''s two-step verification' using errcode = '42501';
    end if;
    return new;
end $$;

-- self sign-up (v5): create an account, confirm its email, approve / decline it
grant insert (pending_approval, email, email_verified, email_token_hash, email_token_expires)
    on public.admin_users to stego_app;
grant update (pending_approval, email, email_verified, email_token_hash, email_token_expires)
    on public.admin_users to stego_app;

-- -------------------------------------------------------------------------------------
-- 3c. Two-step verification (v4): recovery-code table locked down like the others,
--     stego_app may set up / turn off MFA and mark recovery codes used. Never delete.
-- -------------------------------------------------------------------------------------
-- 2. Close the Supabase REST door on the new table, as for the others ------------------
alter table public.mfa_recovery_codes enable row level security;
revoke all on public.mfa_recovery_codes from anon, authenticated;
revoke all on sequence public.mfa_recovery_codes_id_seq from anon, authenticated;

-- 3. What stego_app may do -------------------------------------------------------------
-- admin_users: set up / turn off two-step verification, record the last used time step.
grant update (mfa_enabled, mfa_secret_enc, mfa_pending_enc, mfa_pending_at, mfa_last_step)
    on public.admin_users to stego_app;
grant insert (mfa_enabled, mfa_secret_enc, mfa_pending_enc, mfa_pending_at, mfa_last_step)
    on public.admin_users to stego_app;
-- recovery codes: add new ones, mark one as used. Never delete, never change a hash.
grant select, insert on public.mfa_recovery_codes to stego_app;
grant update (used_at) on public.mfa_recovery_codes to stego_app;
grant usage on sequence public.mfa_recovery_codes_id_seq to stego_app;
drop policy if exists stego_app_all on public.mfa_recovery_codes;
create policy stego_app_all on public.mfa_recovery_codes for all to stego_app using (true) with check (true);

revoke all on function public.protect_admin_accounts() from public, anon, authenticated;

drop trigger if exists protect_admin_accounts on public.admin_users;
create trigger protect_admin_accounts
    before insert or update on public.admin_users
    for each row execute function public.protect_admin_accounts();

commit;

-- -------------------------------------------------------------------------------------
-- 4. Check (read-only). Expected: rls_enabled = true for all 5 tables, and stego_app
--    privileges exactly as granted above.
-- -------------------------------------------------------------------------------------
select c.relname as table_name, c.relrowsecurity as rls_enabled
from pg_class c join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public'
  and c.relname in ('admin_users', 'refresh_tokens', 'revoked_families', 'audit_log', 'mfa_recovery_codes')
order by 1;
