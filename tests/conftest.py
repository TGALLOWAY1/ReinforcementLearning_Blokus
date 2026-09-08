"""Shared pytest configuration."""

import os

import pytest


@pytest.fixture(autouse=True)
def _no_game_log_by_default(monkeypatch):
    """The research profile logs every web game to data/human_games by default
    (webapi/game_log.py). Tests must not write there; tests that exercise the
    log set GAME_LOG_DIR explicitly to a temporary directory."""
    if "GAME_LOG_DIR" not in os.environ:
        monkeypatch.setenv("GAME_LOG_DIR", "")
