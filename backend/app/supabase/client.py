"""
Supabase Client

Centralized, singleton Supabase client initialization.
Loads credentials from environment variables via the Settings
object. Never hardcodes secrets. Reuses connections across
the entire application lifecycle.
"""

from supabase import create_client, Client

from backend.app.config import settings


_client: Client | None = None


def get_supabase_client() -> Client:
    """Return the global Supabase client, creating it on first call.

    The client is initialized once and reused across all requests.
    Credentials are loaded from environment variables.

    Returns:
        Supabase client instance.

    Raises:
        RuntimeError: If SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY
            are not set.
    """
    global _client
    if _client is None:
        if not settings.supabase_url:
            raise RuntimeError(
                "SUPABASE_URL must be set via the SUPABASE_URL environment variable."
            )
        if not settings.supabase_service_role_key:
            raise RuntimeError(
                "SUPABASE_SERVICE_ROLE_KEY must be set via the SUPABASE_SERVICE_ROLE_KEY environment variable."
            )
        _client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
    return _client


def get_supabase_anon_client() -> Client:
    """Return a Supabase client using the anon key for public-facing operations.

    Use this client when RLS policies should be enforced (e.g., user-facing reads).

    Returns:
        Supabase client instance with anon key.

    Raises:
        RuntimeError: If SUPABASE_URL or SUPABASE_ANON_KEY are not set.
    """
    if not settings.supabase_url:
        raise RuntimeError(
            "SUPABASE_URL must be set via the SUPABASE_URL environment variable."
        )
    if not settings.supabase_anon_key:
        raise RuntimeError(
            "SUPABASE_ANON_KEY must be set via the SUPABASE_ANON_KEY environment variable."
        )
    return create_client(
        settings.supabase_url,
        settings.supabase_anon_key,
    )