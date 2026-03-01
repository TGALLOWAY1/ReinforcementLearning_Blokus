"""
test_worker_bridge_save_load.py

Compatibility test between WebWorkerGameBridge.get_state() (the save format)
and WebWorkerGameBridge.load_game() (the load path).

If the game history / save format changes, this test will fail and remind the
developer to update load_game() accordingly.
"""
import json
import os
import random
import sys

import pytest

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BROWSER_PYTHON = os.path.join(PROJECT_ROOT, "browser_python")
# browser_python must come first — worker_bridge and its deps (engine.*, agents.*)
# all expect to be resolved relative to the browser_python directory.
sys.path.insert(0, BROWSER_PYTHON)
sys.path.insert(1, PROJECT_ROOT)

from worker_bridge import WebWorkerGameBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bridge_with_moves(num_moves: int = 8, seed: int = 42) -> WebWorkerGameBridge:
    """Create a bridge, initialise a game, and play `num_moves` random legal moves."""
    random.seed(seed)
    bridge = WebWorkerGameBridge()
    bridge.init_game({
        "game_id": "test-save-load",
        "players": [
            {"player": "RED",    "agent_type": "human", "agent_config": {}},
            {"player": "BLUE",   "agent_type": "human", "agent_config": {}},
            {"player": "GREEN",  "agent_type": "human", "agent_config": {}},
            {"player": "YELLOW", "agent_type": "human", "agent_config": {}},
        ]
    })

    for _ in range(num_moves):
        state = bridge.get_state()
        if state["game_over"]:
            break
        legal = state["legal_moves"]
        if not legal:
            break
        m = random.choice(legal)
        bridge.make_move(m["piece_id"], m["orientation"], m["anchor_row"], m["anchor_col"])

    return bridge


# ---------------------------------------------------------------------------
# Format contract tests
# ---------------------------------------------------------------------------

class TestSaveFormat:
    """Verify the history entries match the documented save format contract."""

    def setup_method(self):
        self.bridge = _make_bridge_with_moves(num_moves=8)
        self.state = self.bridge.get_state()
        self.history = self.state["game_history"]

    def test_history_is_list(self):
        assert isinstance(self.history, list), "game_history must be a list"
        assert len(self.history) > 0, "game_history must not be empty after moves"

    def test_each_entry_has_required_top_level_keys(self):
        required = {"turn_number", "player_to_move", "action", "board_state", "metrics"}
        for i, entry in enumerate(self.history):
            missing = required - set(entry.keys())
            assert not missing, f"Turn {i+1} missing keys: {missing}"

    def test_action_has_required_fields(self):
        required = {"piece_id", "orientation", "anchor_row", "anchor_col"}
        for i, entry in enumerate(self.history):
            action = entry.get("action")
            assert action is not None, f"Turn {i+1} has null action"
            missing = required - set(action.keys())
            assert not missing, f"Turn {i+1} action missing keys: {missing}"

    def test_metrics_has_required_fields(self):
        required = {"corner_count", "frontier_size", "difficult_piece_penalty", "remaining_pieces"}
        for i, entry in enumerate(self.history):
            m = entry.get("metrics", {})
            missing = required - set(m.keys())
            assert not missing, f"Turn {i+1} metrics missing keys: {missing}"

    def test_frontier_metrics_backfilled_in_all_entries(self):
        """All history entries should have frontier_metrics after get_state() is called."""
        frontier_keys = {"frontier_metrics", "frontier_clusters", "piece_lock_risk"}
        for i, entry in enumerate(self.history):
            m = entry.get("metrics", {})
            missing = frontier_keys - set(m.keys())
            assert not missing, (
                f"Turn {i+1} missing frontier fields: {missing}. "
                "Did the backfill logic in get_state() break?"
            )

    def test_player_to_move_is_valid_player_name(self):
        valid = {"RED", "BLUE", "GREEN", "YELLOW"}
        for i, entry in enumerate(self.history):
            assert entry["player_to_move"] in valid, (
                f"Turn {i+1} has invalid player_to_move: {entry['player_to_move']}"
            )


# ---------------------------------------------------------------------------
# Save -> Load round-trip tests
# ---------------------------------------------------------------------------

class TestSaveLoadRoundTrip:
    """
    Verify that history saved by get_state() can be fully replayed by load_game().

    These tests will fail if the save format changes without updating load_game().
    """

    def setup_method(self):
        self.bridge_original = _make_bridge_with_moves(num_moves=8)
        self.original_state = self.bridge_original.get_state()
        self.original_history = self.original_state["game_history"]

    def _load_via_stripped(self):
        """Load using the stripped format that the frontend sends."""
        stripped = [
            {"player_to_move": e["player_to_move"], "action": e["action"]}
            for e in self.original_history
        ]
        bridge2 = WebWorkerGameBridge()
        return bridge2.load_game(stripped)

    def _load_via_full(self):
        """Load using the full history (backward compatibility check)."""
        bridge2 = WebWorkerGameBridge()
        return bridge2.load_game(self.original_history)

    def test_stripped_load_produces_same_move_count(self):
        loaded_state = self._load_via_stripped()
        assert loaded_state["move_count"] == self.original_state["move_count"], (
            f"Loaded move_count {loaded_state['move_count']} != "
            f"original {self.original_state['move_count']}"
        )

    def test_stripped_load_produces_same_board(self):
        loaded_state = self._load_via_stripped()
        assert loaded_state["board"] == self.original_state["board"], (
            "Board after load_game does not match original board. "
            "The save format or orientation mapping may have changed."
        )

    def test_stripped_load_produces_same_scores(self):
        loaded_state = self._load_via_stripped()
        assert loaded_state["scores"] == self.original_state["scores"]

    def test_full_history_load_also_works(self):
        """Full history (with metrics) must still be loadable for backward compat."""
        loaded_state = self._load_via_full()
        assert loaded_state["board"] == self.original_state["board"], (
            "load_game broke when given the full (unstripped) history."
        )

    def test_loaded_game_is_correct_status(self):
        loaded_state = self._load_via_stripped()
        original_over = self.original_state["game_over"]
        assert loaded_state["game_over"] == original_over, (
            f"game_over mismatch: loaded={loaded_state['game_over']}, "
            f"original={original_over}"
        )

    def test_loaded_game_has_expected_game_id(self):
        loaded_state = self._load_via_stripped()
        assert loaded_state["game_id"] == "loaded-game", (
            f"Expected game_id 'loaded-game', got {loaded_state['game_id']!r}"
        )

    def test_loaded_game_history_length_matches(self):
        """After load+get_state, the replayed history should match the original."""
        loaded_state = self._load_via_stripped()
        assert len(loaded_state["game_history"]) == len(self.original_history), (
            f"game_history length mismatch: loaded {len(loaded_state['game_history'])} "
            f"vs original {len(self.original_history)}"
        )


# ---------------------------------------------------------------------------
# JSON serialisability
# ---------------------------------------------------------------------------

class TestJsonSerialisability:
    """Ensure get_state() produces history that can be round-tripped through JSON."""

    def test_game_history_is_json_serialisable(self):
        bridge = _make_bridge_with_moves(num_moves=4)
        state = bridge.get_state()
        # This must not raise
        serialised = json.dumps(state["game_history"])
        assert len(serialised) > 0

    def test_json_round_trip_action_fields_preserved(self):
        bridge = _make_bridge_with_moves(num_moves=4)
        state = bridge.get_state()
        history = state["game_history"]

        serialised = json.dumps(history)
        deserialised = json.loads(serialised)

        for orig, reloaded in zip(history, deserialised):
            assert orig["action"] == reloaded["action"], (
                "action fields changed after JSON round-trip"
            )
            assert orig["player_to_move"] == reloaded["player_to_move"]
