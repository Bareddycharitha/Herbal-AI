"""
Supabase Client Package

Centralized Supabase client initialization and utilities.
The Supabase client is a singleton that is reused across the
entire application lifecycle.
"""

from backend.app.supabase.client import get_supabase_client

__all__ = ["get_supabase_client"]