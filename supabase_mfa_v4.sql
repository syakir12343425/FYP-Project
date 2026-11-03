-- =====================================================================================
-- supabase_mfa_v4.sql  -  add two-step verification (TOTP) to an existing v3 Supabase setup.
-- Run ONCE in the Supabase SQL editor (it runs as the owner). Safe to re-run.
-- Does NOT touch the stego_app password. Needs v3 first (supabase_multiuser_v3.sql).
-- =====================================================================================
begin;

-- 1. New columns on admin_users and the recovery-code table ---------------------------
alter table public.admin_users add column if not exists mfa_enabled boolean not null default false;
alter table public.admin_users add column if not exists mfa_secret_enc varchar(255);
alter table public.admin_users add column if not exists mfa_pending_enc varchar(255);
alter table public.admin_users add column if not exists mfa_pending_at timestamp with time zone;
alter table public.admin_users add column if not exists mfa_last_step integer not null default 0;

create table if not exists public.mfa_recovery_codes (
    id         serial primary key,
    user_id    integer not null references public.admin_users(id) on delete cascade,
    code_hash  varchar(64) not null,
    used_at    timestamp with time zone,
    created_at timestamp with time zone
);
create index if not exists ix_mfa_recovery_codes_user_id on public.mfa_recovery_codes (user_id);
create index if not exists ix_mfa_recovery_codes_code_hash on public.mfa_recovery_codes (code_hash);

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

-- 4. Admin protection trigger, extended -------------------------------------------------
-- New rule: once an ADMIN has two-step verification on, stego_app can neither turn it off nor swap
-- the secret (that needs create_admin.py with owner rights). Setting it up the first time is allowed.
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
    if old.role = 'admin' and old.mfa_enabled and (new.mfa_enabled is distinct from old.mfa_enabled
                                                   or new.mfa_secret_enc is distinct from old.mfa_secret_enc) then
        raise exception 'stego_app may not turn off or replace an admin''s two-step verification' using errcode = '42501';
    end if;
    return new;
end $$;

revoke all on function public.protect_admin_accounts() from public, anon, authenticated;
drop trigger if exists protect_admin_accounts on public.admin_users;
create trigger protect_admin_accounts
    before insert or update on public.admin_users
    for each row execute function public.protect_admin_accounts();

commit;

-- Check: rls_enabled true for all 5 tables, and the new columns present.
select c.relname as table_name, c.relrowsecurity as rls_enabled
from pg_class c join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public'
  and c.relname in ('admin_users', 'refresh_tokens', 'revoked_families', 'audit_log', 'mfa_recovery_codes')
order by 1;
select username, role, mfa_enabled from public.admin_users order by id;
