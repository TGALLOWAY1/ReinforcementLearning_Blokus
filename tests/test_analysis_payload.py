import asyncio
from datetime import datetime, timedelta

from schemas.game_state import GameConfig, AgentType, PlayerConfig, Player
from webapi.app import game_manager, get_game_analysis


def test_analysis_payload_shape_from_memory_game():
    game_id = "analysis-shape-test"
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

    game_manager.games[game_id]["created_at"] = datetime.now() - timedelta(seconds=10)
    game_manager.games[game_id]["updated_at"] = datetime.now()
    game_manager.games[game_id]["move_records"] = [
        {
            "moveIndex": 1,
            "isHuman": False,
            "stats": {"nodesEvaluated": 42, "timeSpentMs": 1000, "maxDepthReached": 2},
        },
        {
            "moveIndex": 2,
            "isHuman": True,
            "stats": {"userMoveTimeMs": 2300},
        },
    ]

    payload = asyncio.run(get_game_analysis(game_id))

    assert "aggregates" in payload
    assert payload["aggregates"]["totalAiNodesEvaluated"] >= 42
    assert "moves" in payload
