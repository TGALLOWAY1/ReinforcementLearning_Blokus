"""Per-move time policy for the natively served Pentobi agent (D-019, D-024).

Pentobi's strength knob is its *level* (a fixed simulation count per move, with
its own internal weighting by move number and an early abort when the best move
cannot change). Wall time is therefore an outcome, not an input. This ledger
adds the controls the human protocol asked for around that:

* a hard per-move cap (default 10 s) that the serving adapter enforces with a
  watchdog and a deterministic fallback move;
* no search at all when a single legal move exists;
* an optional per-game budget: once the remaining budget can no longer fund a
  full-cap move, the level steps down to a fast floor level for the rest of the
  game instead of risking timeouts (the cap itself is never shrunk).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

MIN_LEVEL = 1
MAX_LEVEL = 9
DEFAULT_CAP_MS = 10_000
DEFAULT_FLOOR_LEVEL = 3


@dataclass(frozen=True)
class MoveBudgetDecision:
    level: int
    cap_ms: int
    tier: str
    search: bool
    reasons: List[str] = field(default_factory=list)


class PentobiMoveBudget:
    def __init__(self, level: int, cap_ms: int = DEFAULT_CAP_MS, game_budget_ms: Optional[int] = None,
                 floor_level: int = DEFAULT_FLOOR_LEVEL) -> None:
        level = int(level)
        if not MIN_LEVEL <= level <= MAX_LEVEL:
            raise ValueError(f"level must be in {MIN_LEVEL}..{MAX_LEVEL}, got {level}")
        if int(cap_ms) <= 0:
            raise ValueError("cap_ms must be positive")
        if game_budget_ms is not None and int(game_budget_ms) <= 0:
            raise ValueError("game_budget_ms must be positive when set")
        if not MIN_LEVEL <= int(floor_level) <= level:
            raise ValueError("floor_level must be between 1 and the configured level")
        self.level = level
        self.cap_ms = int(cap_ms)
        self.game_budget_ms = int(game_budget_ms) if game_budget_ms is not None else None
        self.floor_level = int(floor_level)
        self.spent_ms = 0

    @property
    def remaining_ms(self) -> Optional[int]:
        if self.game_budget_ms is None:
            return None
        return max(0, self.game_budget_ms - self.spent_ms)

    def record(self, elapsed_ms: float) -> None:
        self.spent_ms += int(round(elapsed_ms))

    def decide(self, legal_move_count: int, pieces_left: int) -> MoveBudgetDecision:
        if legal_move_count <= 1:
            return MoveBudgetDecision(level=self.level, cap_ms=0, tier="trivial", search=False,
                                      reasons=["one_legal_move"])
        remaining = self.remaining_ms
        if remaining is not None and remaining < self.cap_ms:
            # A cap shrunk to the remainder would guarantee a timeout fallback on
            # the boundary move; step down to the floor level with a full cap.
            reason = "game_budget_exhausted" if remaining <= 0 else "game_budget_below_cap"
            return MoveBudgetDecision(level=self.floor_level, cap_ms=self.cap_ms, tier="budget_exhausted",
                                      search=True, reasons=[reason])
        return MoveBudgetDecision(level=self.level, cap_ms=self.cap_ms, tier="normal", search=True, reasons=[])

    def snapshot(self) -> Dict[str, Any]:
        return {"level": self.level, "capMs": self.cap_ms, "gameBudgetMs": self.game_budget_ms,
                "gameSpentMs": self.spent_ms, "gameRemainingMs": self.remaining_ms}
