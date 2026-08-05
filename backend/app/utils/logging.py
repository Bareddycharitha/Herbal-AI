"""
Logging Configuration

Structured logging setup using structlog with JSON output
for production observability.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor


def add_request_id(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add request ID to log entries if available."""
    # Request ID is added by middleware
    return event_dict


def add_service_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add service context to all log entries."""
    event_dict["service"] = "herbal-ai-api"
    return event_dict


def drop_color_message_key(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Remove color_message key from structlog output."""
    event_dict.pop("color_message", None)
    return event_dict


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """
    Configure structlog for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format (json or text)
    """
    # Standard library logging config
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Shared processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        add_service_context,
        add_request_id,
        drop_color_message_key,
    ]

    if log_format == "json":
        # JSON output for production
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Human-readable output for development
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    # Silence noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)