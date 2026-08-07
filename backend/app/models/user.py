"""
User Model

Clerk-based user profile model.
This module exists for type hints and reference only.
All user data is stored in Supabase PostgreSQL.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class UserProfile:
    """Application-level user profile.

    Mirrors the `profiles` table in Supabase PostgreSQL.
    Uses Clerk user ID (`clerk_user_id`) as the unique identifier.
    Used for type hints and data transfer within the application.
    """

    clerk_user_id: str
    email: str
    full_name: Optional[str] = None
    role: str = "user"
    is_active: bool = True
    created_at: Optional[str] = None
