-- =============================================================================
-- Herbal-AI Supabase Consolidated Database Schema
-- =============================================================================
-- PostgreSQL / Supabase schema for Herbal-AI application.
-- Clerk is used for authentication; Supabase is used as the PostgreSQL database.
-- Run this script in the Supabase SQL Editor to initialize or update all tables.
-- =============================================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- 1. Profiles Table
-- =============================================================================
-- Stores application-level user profiles mapped to Clerk User ID (`clerk_user_id`).
CREATE TABLE IF NOT EXISTS profiles (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    clerk_user_id TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    full_name TEXT,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin', 'researcher')),
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for profiles
CREATE INDEX IF NOT EXISTS idx_profiles_clerk_user_id ON profiles(clerk_user_id);
CREATE INDEX IF NOT EXISTS idx_profiles_email ON profiles(email);
CREATE INDEX IF NOT EXISTS idx_profiles_role ON profiles(role);

-- Auto-update updated_at timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_profiles_updated_at ON profiles;
CREATE TRIGGER trigger_profiles_updated_at
    BEFORE UPDATE ON profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- =============================================================================
-- 2. Skin Disease Prediction History Table
-- =============================================================================
-- Stores image predictions, AI summaries, confidence scores, and herbal suggestions.
CREATE TABLE IF NOT EXISTS prediction_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    profile_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL,
    prediction TEXT NOT NULL,
    confidence NUMERIC(5, 2) NOT NULL DEFAULT 0.0,
    confidence_level TEXT,
    disease_information JSONB DEFAULT '{}'::jsonb,
    recommended_herbs JSONB DEFAULT '[]'::jsonb,
    ai_summary TEXT,
    image_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for prediction_history
CREATE INDEX IF NOT EXISTS idx_prediction_history_profile_id ON prediction_history(profile_id);
CREATE INDEX IF NOT EXISTS idx_prediction_history_created_at ON prediction_history(created_at DESC);


-- =============================================================================
-- 3. Herb Identification History Table
-- =============================================================================
-- Stores botanical plant identification history.
CREATE TABLE IF NOT EXISTS herb_identification_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    profile_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL,
    herb TEXT NOT NULL,
    confidence NUMERIC(5, 2) NOT NULL DEFAULT 0.0,
    confidence_level TEXT,
    herb_information JSONB DEFAULT '{}'::jsonb,
    ai_summary TEXT,
    image_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for herb_identification_history
CREATE INDEX IF NOT EXISTS idx_herb_history_profile_id ON herb_identification_history(profile_id);
CREATE INDEX IF NOT EXISTS idx_herb_history_created_at ON herb_identification_history(created_at DESC);


-- =============================================================================
-- 4. Chat History Table
-- =============================================================================
-- Stores conversations with the AI Dermatology assistant.
CREATE TABLE IF NOT EXISTS chat_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    profile_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL,
    session_id UUID DEFAULT gen_random_uuid(),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    message TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for chat_history
CREATE INDEX IF NOT EXISTS idx_chat_history_profile_id ON chat_history(profile_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(created_at);


-- =============================================================================
-- 5. Generated PDF Reports Table
-- =============================================================================
-- Stores generated clinical PDF reports metadata and file paths.
CREATE TABLE IF NOT EXISTS generated_reports (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    profile_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL,
    prediction_id UUID REFERENCES prediction_history(id) ON DELETE SET NULL,
    report_path TEXT,
    report_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_reports_profile_id ON generated_reports(profile_id);


-- =============================================================================
-- 6. Row Level Security (RLS) Configuration
-- =============================================================================
-- Enable RLS on all tables
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE prediction_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE herb_identification_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE generated_reports ENABLE ROW LEVEL SECURITY;

-- Note: The backend accesses Supabase using the SUPABASE_SERVICE_ROLE_KEY
-- which automatically bypasses RLS for administrative and server-side operations.
-- The policies below govern direct client or authenticated access if enabled.

CREATE POLICY "Service role full access to profiles"
    ON profiles FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role full access to prediction_history"
    ON prediction_history FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role full access to herb_identification_history"
    ON herb_identification_history FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role full access to chat_history"
    ON chat_history FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role full access to generated_reports"
    ON generated_reports FOR ALL
    USING (true)
    WITH CHECK (true);
