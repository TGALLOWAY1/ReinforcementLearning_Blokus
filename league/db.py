"""
SQLite persistence for Elo league data.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional


class LeagueDB:
    def __init__(self, path: str = "league.db"):
        self.path = path
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                type TEXT NOT NULL,
                version TEXT,
                checkpoint_path TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ratings (
                agent_id INTEGER PRIMARY KEY,
                elo REAL NOT NULL,
                FOREIGN KEY(agent_id) REFERENCES agents(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent1_id INTEGER NOT NULL,
                agent2_id INTEGER NOT NULL,
                result REAL NOT NULL,
                seed INTEGER,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(agent1_id) REFERENCES agents(id) ON DELETE CASCADE,
                FOREIGN KEY(agent2_id) REFERENCES agents(id) ON DELETE CASCADE
            );
            """
        )
        self._conn.commit()

    def add_agent(
        self,
        name: str,
        agent_type: str,
        version: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        initial_elo: float = 1200.0,
    ) -> int:
        now = datetime.utcnow().isoformat()
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO agents (name, type, version, checkpoint_path, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, agent_type, version, checkpoint_path, now),
        )
        self._conn.commit()
        agent_id = self.get_agent_id(name)
        if agent_id is not None:
            cursor.execute(
                "INSERT OR IGNORE INTO ratings (agent_id, elo) VALUES (?, ?)",
                (agent_id, initial_elo),
            )
            self._conn.commit()
        return agent_id

    def get_agent_id(self, name: str) -> Optional[int]:
        cursor = self._conn.execute("SELECT id FROM agents WHERE name = ?", (name,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_agent(self, name: str) -> Optional[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT id, name, type, version, checkpoint_path FROM agents WHERE name = ?",
            (name,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "type": row[2],
            "version": row[3],
            "checkpoint_path": row[4],
        }

    def record_match(self, agent1_id: int, agent2_id: int, result: float, seed: Optional[int]) -> None:
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """
            INSERT INTO matches (agent1_id, agent2_id, result, seed, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (agent1_id, agent2_id, result, seed, now),
        )
        self._conn.commit()

    def update_rating(self, agent_id: int, new_elo: float) -> None:
        self._conn.execute(
            "UPDATE ratings SET elo = ? WHERE agent_id = ?",
            (new_elo, agent_id),
        )
        self._conn.commit()

    def get_rating(self, agent_id: int) -> float:
        cursor = self._conn.execute("SELECT elo FROM ratings WHERE agent_id = ?", (agent_id,))
        row = cursor.fetchone()
        return float(row[0]) if row else 1200.0

    def leaderboard(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = """
            SELECT agents.name, agents.type, agents.version, ratings.elo
            FROM ratings
            JOIN agents ON agents.id = ratings.agent_id
            ORDER BY ratings.elo DESC
        """
        if limit:
            query += " LIMIT ?"
            rows = self._conn.execute(query, (limit,)).fetchall()
        else:
            rows = self._conn.execute(query).fetchall()
        return [
            {"name": row[0], "type": row[1], "version": row[2], "elo": row[3]}
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()
