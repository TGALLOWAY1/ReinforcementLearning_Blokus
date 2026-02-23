"""
Factory helpers for deploy-mode gameplay agent adapters.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agents.gameplay_fast_mcts import GameplayFastMCTSAgent
from agents.gameplay_protocol import GameplayAgentProtocol
from schemas.game_state import AgentType


def build_deploy_gameplay_agent(
    agent_type: AgentType,
    agent_config: Optional[Dict[str, Any]] = None,
) -> Optional[GameplayAgentProtocol]:
    """
    Build gameplay adapters used by APP_PROFILE=deploy.

    Human players are returned as ``None`` because websocket drives those turns.
    """
    cfg = dict(agent_config or {})
    if agent_type == AgentType.HUMAN:
        return None
    if agent_type == AgentType.MCTS:
        return GameplayFastMCTSAgent(
            iterations=int(cfg.get("iterations", 5000)),
            exploration_constant=float(cfg.get("exploration_constant", 1.414)),
            seed=cfg.get("seed"),
        )
    raise ValueError(f"Unsupported deploy agent type: {agent_type}")


def is_gameplay_adapter(agent: Any) -> bool:
    return callable(getattr(agent, "choose_move", None))
