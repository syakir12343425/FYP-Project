-- =====================================================================================
-- supabase_signup_v5.sql  -  self sign-up with email confirmation, on top of v4 (two-step verification).
-- Run ONCE in the Supabase SQL editor (it runs as the owner). Safe to re-run.
-- Does NOT touch the stego_app password.
-- =====================================================================================
begin;

-- 1. New columns. Accounts that exist before this runs need no email confirmation (first run only).
do $$
begin
    if not exists (select 1 from information_schema.columns
                   where table_schema = 'public' and table_name = 'admin_users' and column_name = 'email_verified') then
        alter table public.admin_users add column email_verified boolean not null default false;
        update public.admin_users set email_verified = true;
    end if;
end $$;
alter table public.admin_users add column if not exists pending_approval boolean not null default false;
alter table public.admin_users add column if not exists email varchar(254);
alter table public.admin_users add column if not exists email_token_hash varchar(64);
alter table public.admin_users add column if not exists email_token_expires timestamp with time zone;
create unique index if not exists ix_admin_users_email on public.admin_users (email);
create index if not exists ix_admin_users_email_token_hash on public.admin_users (email_token_hash);

-- 2. What stego_app may do: create a sign-up (role 'user' only, enforced by the trigger), confirm the email,
--    send a new link, release an address held by an expired unconfirmed sign-up, approve / decline.
grant insert (pending_approval, email, email_verified, email_token_hash, email_token_expires)
    on public.admin_users to stego_app;
grant update (pending_approval, email, email_verified, email_token_hash, email_token_expires)
    on public.admin_users to stego_app;

-- 3. Trigger, extended: stego_app may not change an ADMIN's email address or its confirmed flag.
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
revoke all on function public.protect_admin_accounts() from public, anon, authenticated;

commit;

-- Check: existing accounts confirmed, nobody waiting.
select username, role, email_verified, mfa_enabled from public.admin_users order by id;
