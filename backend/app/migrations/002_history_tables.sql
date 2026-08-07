-- ============================================================
-- History Tables (Prepared for Future Use)
-- ============================================================
-- These tables are schema-only at this stage. No API endpoints
-- are implemented yet. They are designed to support future
-- history features while referencing Supabase PostgreSQL
-- profiles via the clerk_user_id column.

-- ============================================================
-- Prediction History
-- ============================================================
create table if not exists prediction_history (
    id uuid default gen_random_uuid() primary key,
    profile_id text references profiles(clerk_user_id) on delete cascade not null,
    image_path text,
    prediction jsonb not null,
    confidence numeric(5, 2),
    confidence_level text,
    top_predictions jsonb,
    disease_information jsonb,
    recommended_herbs jsonb,
    ai_summary text,
    created_at timestamptz not null default now()
);

create index if not exists idx_prediction_history_profile_id on prediction_history (profile_id);
create index if not exists idx_prediction_history_created_at on prediction_history (created_at);

-- ============================================================
-- Herb Identification History
-- ============================================================
create table if not exists herb_identification_history (
    id uuid default gen_random_uuid() primary key,
    profile_id text references profiles(clerk_user_id) on delete cascade not null,
    image_path text,
    prediction jsonb not null,
    confidence numeric(5, 2),
    confidence_level text,
    top_predictions jsonb,
    herb_information jsonb,
    ai_summary text,
    created_at timestamptz not null default now()
);

create index if not exists idx_herb_history_profile_id on herb_identification_history (profile_id);
create index if not exists idx_herb_history_created_at on herb_identification_history (created_at);

-- ============================================================
-- Chat History
-- ============================================================
create table if not exists chat_history (
    id uuid default gen_random_uuid() primary key,
    profile_id text references profiles(clerk_user_id) on delete cascade not null,
    session_id uuid,
    message text not null,
    role text not null check (role in ('user', 'assistant', 'system')),
    metadata jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_chat_history_profile_id on chat_history (profile_id);
create index if not exists idx_chat_history_session_id on chat_history (session_id);
create index if not exists idx_chat_history_created_at on chat_history (created_at);

-- ============================================================
-- Generated Reports
-- ============================================================
create table if not exists generated_reports (
    id uuid default gen_random_uuid() primary key,
    profile_id text references profiles(clerk_user_id) on delete cascade not null,
    report_path text,
    prediction_data jsonb,
    disease_info jsonb,
    herbal_recommendations jsonb,
    summary text,
    gradcam_image text,
    created_at timestamptz not null default now()
);

create index if not exists idx_reports_profile_id on generated_reports (profile_id);
create index if not exists idx_reports_created_at on generated_reports (created_at);

-- ============================================================
-- Saved Images
-- ============================================================
create table if not exists saved_images (
    id uuid default gen_random_uuid() primary key,
    profile_id text references profiles(clerk_user_id) on delete cascade not null,
    image_path text not null,
    image_type text not null check (image_type in ('skin', 'herb', 'gradcam', 'report')),
    metadata jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_saved_images_profile_id on saved_images (profile_id);
create index if not exists idx_saved_images_type on saved_images (image_type);

-- ============================================================
-- User Preferences
-- ============================================================
create table if not exists user_preferences (
    id uuid default gen_random_uuid() primary key,
    profile_id text references profiles(clerk_user_id) on delete cascade not null,
    preferences jsonb not null default '{}',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (profile_id)
);

create index if not exists idx_user_preferences_profile_id on user_preferences (profile_id);

-- Keep updated_at current
create or replace function update_user_preferences_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists trigger_update_user_preferences_updated_at on user_preferences;
create trigger trigger_update_user_preferences_updated_at
    before update on user_preferences
    for each row
    execute function update_user_preferences_updated_at();

-- ============================================================
-- Audit Logs
-- ============================================================
create table if not exists audit_logs (
    id uuid default gen_random_uuid() primary key,
    profile_id text references profiles(clerk_user_id) on delete cascade not null,
    action text not null,
    entity_type text,
    entity_id text,
    details jsonb,
    ip_address text,
    user_agent text,
    created_at timestamptz not null default now()
);

create index if not exists idx_audit_logs_profile_id on audit_logs (profile_id);
create index if not exists idx_audit_logs_action on audit_logs (action);
create index if not exists idx_audit_logs_created_at on audit_logs (created_at);
