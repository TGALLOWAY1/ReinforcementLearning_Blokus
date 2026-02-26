"""
Database module for Blokus RL Web API.

This module provides MongoDB connection and models for persistent storage.
"""

from .models import EvaluationRun, TrainingRun
from .mongo import close_mongo_connection, connect_to_mongo, get_database

__all__ = [
    "connect_to_mongo",
    "close_mongo_connection",
    "get_database",
    "TrainingRun",
    "EvaluationRun",
]

