import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";

// Supabase client for database operations only.
// Authentication is handled by Clerk, not Supabase Auth.
// This client is used for RLS-enabled reads and admin writes.

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
