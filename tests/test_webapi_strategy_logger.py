"""
Test StrategyLogger wiring for live WebAPI games.

When ENABLE_STRATEGY_LOGGER is unset (default), logging is disabled.
When set to true, live games write steps.jsonl and results.jsonl.
"""

import json
import os

# Ensure we can import webapi
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStrategyLoggerConfig(unittest.TestCase):
    """Test config module."""

    def test_default_disabled(self):
        """By default, strategy logger is disabled."""
        orig = os.environ.pop("ENABLE_STRATEGY_LOGGER", None)
        try:
            from webapi.strategy_logger_config import is_strategy_logger_enabled
            self.assertFalse(is_strategy_logger_enabled())
        finally:
            if orig is not None:
                os.environ["ENABLE_STRATEGY_LOGGER"] = orig

    def test_log_dir_override(self):
        """STRATEGY_LOG_DIR can override default."""
        from webapi.strategy_logger_config import (
            get_strategy_log_dir,
            get_strategy_log_dir_for_game,
        )
        d = get_strategy_log_dir()
        self.assertIn("logs", d)
        sub = get_strategy_log_dir_for_game(d, "game-123")
        self.assertTrue(sub.endswith("game-123") or "game-123" in sub)


class TestStrategyLoggerWiring(unittest.TestCase):
    """Test that _log_step and _log_game_end work when enabled."""

    def test_log_step_with_enabled_logger(self):
        """_log_step writes to steps.jsonl when logger is enabled."""
        orig = os.environ.get("ENABLE_STRATEGY_LOGGER")
        try:
            os.environ["ENABLE_STRATEGY_LOGGER"] = "true"
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["STRATEGY_LOG_DIR"] = tmp
                from importlib import reload

                import webapi.strategy_logger_config as cfg
                reload(cfg)

                from schemas.game_state import (
                    AgentType,
                    GameConfig,
                    GameStatus,
                    Player,
                    PlayerConfig,
                )
                from webapi.app import GameManager

                config = GameConfig(
                    players=[
                        PlayerConfig(player=Player.RED, agent_type=AgentType.RANDOM),
                        PlayerConfig(player=Player.BLUE, agent_type=AgentType.RANDOM),
                    ],
                    auto_start=False,  # Avoid asyncio.create_task (no event loop in test)
                )
                game_manager = GameManager()
                game_id = game_manager.create_game(config)
                game_manager.games[game_id]["status"] = GameStatus.IN_PROGRESS
                game_manager._init_strategy_logger(game_id)
                game_data = game_manager.games[game_id]
                game = game_data["game"]

                # Make one legal move (RED's first move - corner)
                legal = game.get_legal_moves(game.get_current_player())
                self.assertGreater(len(legal), 0, "Need at least one legal move")
                move = legal[0]
                player = game.get_current_player()
                state_before = game.get_board_copy()
                success = game.make_move(move, player)
                self.assertTrue(success)

                game_manager._log_step(
                    game_id, 0, player.value, state_before, move, game.board
                )

                # Check steps.jsonl exists and has entry
                from webapi.strategy_logger_config import get_strategy_log_dir_for_game
                log_dir = get_strategy_log_dir_for_game(tmp, game_id)
                steps_path = os.path.join(log_dir, "steps.jsonl")
                self.assertTrue(os.path.exists(steps_path), f"Expected {steps_path}")
                with open(steps_path) as f:
                    lines = [l for l in f if l.strip()]
                self.assertGreater(len(lines), 0)
                entry = json.loads(lines[0])
                self.assertEqual(entry["game_id"], game_id)
                self.assertEqual(entry["turn_index"], 0)
                self.assertEqual(entry["player_id"], player.value)
                self.assertIn("action", entry)
                self.assertIn("metrics", entry)
        finally:
            if orig is not None:
                os.environ["ENABLE_STRATEGY_LOGGER"] = orig
            else:
                os.environ.pop("ENABLE_STRATEGY_LOGGER", None)
            os.environ.pop("STRATEGY_LOG_DIR", None)

    def test_log_step_disabled_is_noop(self):
        """When logger disabled, _log_step does not crash."""
        orig = os.environ.pop("ENABLE_STRATEGY_LOGGER", None)
        try:
            from schemas.game_state import AgentType, GameConfig, Player, PlayerConfig
            from webapi.app import GameManager

            config = GameConfig(
                players=[
                    PlayerConfig(player=Player.RED, agent_type=AgentType.RANDOM),
                    PlayerConfig(player=Player.BLUE, agent_type=AgentType.RANDOM),
                ],
                auto_start=False,
            )
            game_manager = GameManager()
            game_id = game_manager.create_game(config)
            game = game_manager.games[game_id]["game"]
            legal = game.get_legal_moves(game.get_current_player())
            self.assertGreater(len(legal), 0)
            move = legal[0]
            player = game.get_current_player()
            state_before = game.get_board_copy()
            game.make_move(move, player)
            # No logger - should not crash
            game_manager._log_step(game_id, 0, player.value, state_before, move, game.board)
        finally:
            if orig is not None:
                os.environ["ENABLE_STRATEGY_LOGGER"] = orig


if __name__ == "__main__":
    unittest.main()
