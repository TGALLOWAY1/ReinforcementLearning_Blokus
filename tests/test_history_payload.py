import asyncio
from datetime import datetime, timedelta

from schemas.game_state import AgentType, GameConfig, Player, PlayerConfig
from webapi.app import GameStatus, game_manager, get_history


def test_history_payload_from_memory_finished_games():
    game_id = "history-shape-test"
    config = GameConfig(
        game_id=game_id,
        players=[
            PlayerConfig(player=Player.RED, agent_type=AgentType.HUMAN),
            PlayerConfig(player=Player.BLUE, agent_type=AgentType.MCTS, agent_config={"time_budget_ms": 1000}),
        ],
        auto_start=False,
    )
    if game_id not in game_manager.games:
        game_manager.create_game(config)

    g = game_manager.games[game_id]
    g["created_at"] = datetime.now() - timedelta(seconds=5)
    g["updated_at"] = datetime.now()
    g["status"] = GameStatus.FINISHED
    g["winner"] = "RED"
    g["move_records"] = [{"isHuman": False, "stats": {"nodesEvaluated": 10, "timeSpentMs": 100}}]

    payload = asyncio.run(get_history())
    assert "games" in payload
    assert any(item["game_id"] == game_id for item in payload["games"])
