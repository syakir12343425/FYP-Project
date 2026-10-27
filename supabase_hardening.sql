-- Run in Supabase SQL editor AFTER the server has created the tables (first start).
-- Supabase exposes the public schema through an auto-generated REST API (PostgREST) using the public "anon" key.
-- Our app talks to Postgres directly, so lock the REST door completely:
alter table public.admin_users    enable row level security;
alter table public.refresh_tokens enable row level security;
alter table public.audit_log      enable row level security;
-- With RLS on and NO policies, anon/authenticated roles can read nothing. Our server connects as the
-- postgres owner role (bypasses RLS), so it keeps working.
revoke all on public.admin_users, public.refresh_tokens, public.audit_log from anon, authenticated;
