-- ============================================================
-- Profile Table
-- ============================================================
-- Stores application-level user profiles linked to Clerk
-- authentication. The `clerk_user_id` is the unique
-- identifier from Clerk's `sub` claim.
-- Supabase is used ONLY as the database; Supabase Auth
-- is NOT used.

create table if not exists profiles (
    id uuid default gen_random_uuid() primary key,
    clerk_user_id text unique not null,
    email text unique not null,
    full_name text,
    role text not null default 'user' check (role in ('user', 'admin', 'researcher')),
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- Indexes for common lookups
create index if not exists idx_profiles_email on profiles (email);
create index if not exists idx_profiles_role on profiles (role);
create index if not exists idx_profiles_is_active on profiles (is_active);
create index if not exists idx_profiles_clerk_user_id on profiles (clerk_user_id);

-- Keep updated_at current on profile changes
create or replace function update_profile_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists trigger_update_profile_updated_at on profiles;
create trigger trigger_update_profile_updated_at
    before update on profiles
    for each row
    execute function update_profile_updated_at();

-- ============================================================
-- Row Level Security
-- ============================================================
-- Enable RLS on profiles. Users can read their own profile
-- and admins can read all profiles.

alter table profiles enable row level security;

-- Policy: users can read their own profile
create policy "Users can view own profile"
    on profiles for select
    using (clerk_user_id = current_setting('app.current_user_id', true));

-- Policy: admins can read all profiles
create policy "Admins can view all profiles"
    on profiles for select
    using (
        exists (
            select 1 from profiles
            where clerk_user_id = current_setting('app.current_user_id', true)
            and role = 'admin'
        )
    );

-- Policy: users can update their own profile
create policy "Users can update own profile"
    on profiles for update
    using (clerk_user_id = current_setting('app.current_user_id', true));

-- Policy: admins can update any profile
create policy "Admins can update all profiles"
    on profiles for update
    using (
        exists (
            select 1 from profiles
            where clerk_user_id = current_setting('app.current_user_id', true)
            and role = 'admin'
        )
    );

-- Policy: users can delete their own profile
create policy "Users can delete own profile"
    on profiles for delete
    using (clerk_user_id = current_setting('app.current_user_id', true));
