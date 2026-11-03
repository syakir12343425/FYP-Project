-- =====================================================================================
-- supabase_multiuser_v3.sql  -  upgrade an EXISTING v2 Supabase setup to multi-user accounts.
-- Run ONCE in the Supabase SQL editor (which runs as the owner, so no owner URL in .env is needed).
-- Step 0 adds the new columns (same as tools/migrate_users.py); the rest gives stego_app the new grants
-- and installs the admin-protection trigger. It does NOT touch the stego_app password (no SCRAM line).
-- Safe to re-run. (A fresh setup uses supabase_hardening.sql, which already contains steps 1-2.)
-- =====================================================================================
begin;

-- -------------------------------------------------------------------------------------
-- 0. New columns. Every account that exists before this runs was made by create_admin.py,
--    so it becomes role 'admin' (only on the first run, when the role column is new).
-- -------------------------------------------------------------------------------------
do $$
begin
    if not exists (select 1 from information_schema.columns
                   where table_schema = 'public' and table_name = 'admin_users' and column_name = 'role') then
        alter table public.admin_users add column role varchar(16) not null default 'user';
        update public.admin_users set role = 'admin';
    end if;
end $$;
alter table public.admin_users add column if not exists disabled boolean not null default false;
alter table public.admin_users add column if not exists must_change_password boolean not null default false;
alter table public.admin_users add column if not exists last_login_at timestamp with time zone;

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
                               or new.must_change_password is distinct from old.must_change_password) then
        raise exception 'stego_app may not change admin accounts' using errcode = '42501';
    end if;
    return new;
end $$;

revoke all on function public.protect_admin_accounts() from public, anon, authenticated;

drop trigger if exists protect_admin_accounts on public.admin_users;
create trigger protect_admin_accounts
    before insert or update on public.admin_users
    for each row execute function public.protect_admin_accounts();

commit;

-- Check: expect your admin account(s) with role = admin, and the trigger listed.
select username, role, disabled, must_change_password from public.admin_users order by id;
select tgname as trigger_installed from pg_trigger where tgname = 'protect_admin_accounts';
