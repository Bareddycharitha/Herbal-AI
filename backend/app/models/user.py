"""
SQLAlchemy User Model

ORM model for the users table in PostgreSQL.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, Index

from backend.app.database.base import Base


def _utcnow() -> datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class User(Base):
    """SQLAlchemy ORM model for the users table."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(Text, nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(String(20), nullable=False, default="user")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        Index("idx_users_email", "email", unique=True),
        Index("idx_users_role", "role"),
        Index("idx_users_is_active", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email!r}, role={self.role!r})>"
