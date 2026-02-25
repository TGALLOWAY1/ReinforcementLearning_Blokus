"""
Strategy logger configuration for live WebAPI games.

Logging is optional and disabled by default to avoid breaking deployment
(Vercel/hosted environments may have read-only or ephemeral filesystems).

Set ENABLE_STRATEGY_LOGGER=true to enable.
Set STRATEGY_LOG_DIR to override log directory (default: logs/webapi_analytics).
"""

import os


def is_strategy_logger_enabled() -> bool:
    """True if ENABLE_STRATEGY_LOGGER is set to a truthy value."""
    raw = os.getenv("ENABLE_STRATEGY_LOGGER", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def get_strategy_log_dir() -> str:
    """Log directory for StrategyLogger. Safe fallback for Vercel/hosted."""
    return os.getenv("STRATEGY_LOG_DIR", "logs/webapi_analytics").strip() or "logs/webapi_analytics"


def get_strategy_log_dir_for_game(base_dir: str, game_id: str) -> str:
    """
    Per-game log directory. Each game writes to its own subdir to avoid
    file contention and allow easy cleanup.
    """
    # Sanitize game_id for filesystem (alphanumeric, underscore, hyphen)
    safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in game_id)
    return os.path.join(base_dir, safe_id)
