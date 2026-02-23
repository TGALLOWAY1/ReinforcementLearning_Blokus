"""
Application profile helpers for research vs deploy runtime surfaces.
"""

from __future__ import annotations

import os


APP_PROFILE_RESEARCH = "research"
APP_PROFILE_DEPLOY = "deploy"
_VALID_PROFILES = {APP_PROFILE_RESEARCH, APP_PROFILE_DEPLOY}


def get_app_profile() -> str:
    """
    Return the normalized app profile.

    Defaults to ``research`` when unset or invalid.
    """
    raw = (os.getenv("APP_PROFILE") or APP_PROFILE_RESEARCH).strip().lower()
    if raw not in _VALID_PROFILES:
        return APP_PROFILE_RESEARCH
    return raw


def is_deploy_profile() -> bool:
    """True when APP_PROFILE resolves to deploy."""
    return get_app_profile() == APP_PROFILE_DEPLOY
