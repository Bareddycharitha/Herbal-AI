"""Alembic environment configuration for async SQLAlchemy."""

import asyncio
import os
import sys
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection

from alembic import context

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the database URL from the application settings
from backend.app.config import settings

config.set_main_option(
    "sqlalchemy.url",
    settings.database_url.replace("+asyncpg", ""),
)

# target_metadata
from backend.app.database.base import Base

target_metadata = Base.metadata


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    from sqlalchemy.ext.asyncio import async_engine_from_config

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    # Offline mode: use synchronous engine
    from sqlalchemy import create_engine

    url = config.get_main_option("sqlalchemy.url")
    engine = create_engine(url, poolclass=pool.NullPool)

    with engine.connect() as connection:
        do_run_migrations(connection)
else:
    run_migrations_online()
