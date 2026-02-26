from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from agents.fast_mcts_agent import FastMCTSAgent
from engine.bitboard import coord_to_bit
from engine.board import Board, Player
from engine.move_generator import Move


class ThinkRequest(BaseModel):
    gameState: Dict[str, Any]
    legalMoves: List[Dict[str, Any]]
    timeBudgetMs: int


app = FastAPI(title="Blokus Engine Service")


@app.get('/health')
def health():
    return {"ok": True}


def _rebuild_board(game_state: Dict[str, Any]) -> Board:
    board = Board()
    grid = np.array(game_state.get("board", board.grid.tolist()), dtype=int)
    board.grid = grid
    board.move_count = int(game_state.get("move_count", 0))
    board.current_player = Player[game_state.get("current_player", "RED")]

    pieces_used = game_state.get("pieces_used", {})
    for player in Player:
        used = set(pieces_used.get(player.name, []))
        board.player_pieces_used[player] = used
        board.player_first_move[player] = len(used) == 0
        board.player_frontiers[player] = board._compute_full_frontier(player) if used else {(board.player_start_corners[player].row, board.player_start_corners[player].col)}

    board.occupied_bits = 0
    board.player_bits = {player: 0 for player in Player}
    for r in range(board.SIZE):
        for c in range(board.SIZE):
            v = int(board.grid[r, c])
            if v > 0:
                bit = coord_to_bit(r, c)
                board.occupied_bits |= bit
                board.player_bits[Player(v)] |= bit
    return board


@app.post('/think')
def think(payload: ThinkRequest):
    board = _rebuild_board(payload.gameState)
    player = Player[payload.gameState.get("current_player", "RED")]
    legal_moves = [Move(piece_id=m['piece_id'], orientation=m['orientation'], anchor_row=m['anchor_row'], anchor_col=m['anchor_col']) for m in payload.legalMoves]

    agent = FastMCTSAgent(iterations=5000, time_limit=max(payload.timeBudgetMs, 1) / 1000.0)
    result = agent.think(board, player, legal_moves, payload.timeBudgetMs)
    move = result["move"]
    stats = result.get("stats") or {}

    return {
        "move": None if move is None else {
            "piece_id": move.piece_id,
            "orientation": move.orientation,
            "anchor_row": move.anchor_row,
            "anchor_col": move.anchor_col,
        },
        "stats": stats,
    }
