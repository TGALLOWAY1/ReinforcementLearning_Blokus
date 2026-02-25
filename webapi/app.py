"""
FastAPI application for Blokus RL Web API

Framework: FastAPI (Python async web framework)
MongoDB Module: webapi.db.mongo (provides centralized MongoDB connection)
"""

import asyncio
import json
import uuid
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import urllib.request
import urllib.error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.game import BlokusGame
from engine.board import Player as EnginePlayer
from engine.move_generator import Move as EngineMove
from engine.mobility_metrics import compute_player_mobility_metrics
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from mcts.mcts_agent import MCTSAgent
from agents.fast_mcts_agent import FastMCTSAgent
from schemas.game_state import (
    GameConfig, GameState, GameStatus, Player, AgentType, Move, StateUpdate,
    MoveRequest, MoveResponse, GameCreateResponse, AgentInfo, ErrorResponse
)
from webapi.deploy_validation import (
    DEPLOY_TIME_BUDGET_CAP_MS,
    normalize_deploy_game_config,
)
from webapi.gameplay_agent_factory import build_deploy_gameplay_agent, is_gameplay_adapter
from webapi.profile import APP_PROFILE_DEPLOY, APP_PROFILE_RESEARCH, get_app_profile
from webapi.routes_gameplay import register_gameplay_routes
from webapi.routes_research import register_research_routes
from webapi.strategy_logger_config import (
    is_strategy_logger_enabled,
    get_strategy_log_dir,
    get_strategy_log_dir_for_game,
)

try:
    # MongoDB connection module (webapi/db/mongo.py)
    from webapi.db.mongo import connect_to_mongo, close_mongo_connection, get_database
except Exception as mongo_import_error:  # pragma: no cover - deploy runtime may omit mongo deps
    async def connect_to_mongo() -> None:
        raise RuntimeError(
            f"MongoDB dependencies unavailable: {mongo_import_error}"
        )

    async def close_mongo_connection() -> None:
        return None

    def get_database():
        raise RuntimeError(
            f"MongoDB dependencies unavailable: {mongo_import_error}"
        )


class GameManager:
    """Manages active games and their state."""
    
    def __init__(self, app_profile: Optional[str] = None):
        self.app_profile = app_profile or get_app_profile()
        self.games: Dict[str, Dict[str, Any]] = {}
        self.agent_instances: Dict[str, Any] = {}
        self._strategy_loggers: Dict[str, Any] = {}  # game_id -> StrategyLogger
    
    def create_game(self, config: GameConfig) -> str:
        """Create a new game."""
        game_id = config.game_id or str(uuid.uuid4())
        
        if game_id in self.games:
            raise HTTPException(status_code=400, detail="Game ID already exists")
        
        # Create game instance
        game = BlokusGame()
        
        # Store game data
        now = datetime.now()
        self.games[game_id] = {
            'game': game,
            'config': config,
            'status': GameStatus.WAITING,
            'created_at': now,
            'updated_at': now,
            'move_records': [],
            'last_turn_started_at': time.perf_counter(),
            'last_turn_player': game.get_current_player(),
            'winner': None,
            'configured_players': {self._convert_player(player_cfg.player) for player_cfg in config.players},
        }
        
        # Initialize agent instances
        self._initialize_agents(game_id, config)
        
        # Start game if auto_start is True
        if config.auto_start:
            self._start_game(game_id)
        
        logger.info(f"Game created: {game_id}")
        return game_id
    
    def _initialize_agents(self, game_id: str, config: GameConfig):
        """Initialize agent instances for the game."""
        agents = {}
        for player_config in config.players:
            player = self._convert_player(player_config.player)
            agent_type = player_config.agent_type
            agent_config = player_config.agent_config or {}

            if self.app_profile == APP_PROFILE_DEPLOY:
                agents[player] = build_deploy_gameplay_agent(agent_type, agent_config)
                continue
            
            if agent_type == AgentType.RANDOM:
                agents[player] = RandomAgent()
            elif agent_type == AgentType.HEURISTIC:
                agents[player] = HeuristicAgent()
            elif agent_type == AgentType.MCTS:
                budget_ms = int(agent_config.get('time_budget_ms', 1000))
                agents[player] = FastMCTSAgent(
                    iterations=5000,
                    time_limit=max(budget_ms, 1) / 1000.0,
                    exploration_constant=1.414
                )
            elif agent_type == AgentType.HUMAN:
                agents[player] = None  # Human players don't need agents
            else:
                raise HTTPException(status_code=400, detail=f"Unknown agent type: {agent_type}")
        
        self.agent_instances[game_id] = agents
    
    def _all_configured_players_blocked(self, game_data: Dict[str, Any]) -> bool:
        """True if all configured players have no legal moves."""
        game = game_data['game']
        configured_players = game_data.get('configured_players', set(EnginePlayer))
        if not configured_players:
            return True
        for pl in configured_players:
            if game.get_legal_moves(pl):
                return False
        return True

    def _start_game(self, game_id: str):
        """Start the game."""
        if game_id not in self.games:
            raise HTTPException(status_code=404, detail="Game not found")
        
        self.games[game_id]['status'] = GameStatus.IN_PROGRESS
        self.games[game_id]['updated_at'] = datetime.now()
        
        self._init_strategy_logger(game_id)
        
        # In serverless/polling mode, turns are advanced via HTTP endpoints 
        # instead of a background asyncio task.
    
    async def advance_turn(self, game_id: str) -> MoveResponse:
        """Advance the turn by letting the current agent make a move."""
        if game_id not in self.games:
            raise HTTPException(status_code=404, detail="Game not found")
            
        game_data = self.games[game_id]
        if game_data['status'] != GameStatus.IN_PROGRESS:
            return MoveResponse(
                success=False,
                message="Game is not in progress",
                game_state=self._get_game_state(game_id)
            )
            
        game = game_data['game']
        
        if game.is_game_over() or self._all_configured_players_blocked(game_data):
            logger.info(f"Game {game_id} is over")
            await self._end_game(game_id)
            return MoveResponse(
                success=True,
                message="Game is over",
                game_state=self._get_game_state(game_id)
            )
            
        current_player = game.get_current_player()
        agents = self.agent_instances.get(game_id, {})
        agent = agents.get(current_player)
        
        configured_players = game_data.get('configured_players', set(EnginePlayer))
        if current_player not in configured_players:
            game.board._update_current_player()
            return MoveResponse(
                success=True,
                message="Skipped inactive player",
                game_state=self._get_game_state(game_id)
            )

        if agent is not None:
            # Agent move
            await self._make_agent_move(game_id, current_player, agent)
            
            # Check if game is over after move
            if game.is_game_over() or self._all_configured_players_blocked(game_data):
                await self._end_game(game_id)
                
            return MoveResponse(
                success=True,
                message="Agent moved",
                game_state=self._get_game_state(game_id)
            )
        else:
            return MoveResponse(
                success=False,
                message="Waiting for human move",
                game_state=self._get_game_state(game_id)
            )
    
    async def _make_agent_move(self, game_id: str, player: EnginePlayer, agent: Any):
        """
        Make a move for an agent.
        
        Behavior:
        - If agent has no legal moves, the turn is passed automatically.
        - If agent times out or raises an exception, the turn is passed (safer than random fallback).
        - Only if agent successfully returned a move will it be placed on the board.
        
        This ensures predictable, debuggable behavior when agents fail.
        """
        start_total = time.perf_counter()
        try:
            logger.info(f"AGENT MOVE START: game_id={game_id}, player={player.name}, agent={type(agent).__name__}")
            game_data = self.games[game_id]
            game = game_data['game']
            
            # Get legal moves
            start_legal = time.perf_counter()
            legal_moves = game.get_legal_moves(player)
            end_legal = time.perf_counter()
            legal_time = end_legal - start_legal
            logger.info(f"AGENT MOVE legal_moves: {len(legal_moves)} moves found in {legal_time:.4f}s for player={player.name}")
            
            if not legal_moves:
                # Player cannot move, skip turn
                logger.info(f"AGENT MOVE: No legal moves for {player.name}, skipping turn")
                self._record_pass(game_id, player, agent_type=type(agent).__name__, reason="no_moves")
                game.board._update_current_player()
                game._check_game_over()  # No piece placed, so make_move never ran this
                return
            
            # Get agent's move with timeout / budget
            start_agent = time.perf_counter()
            player_config = next((p for p in game_data['config'].players if self._convert_player(p.player) == player), None)
            budget_ms = int((player_config.agent_config or {}).get('time_budget_ms', 1000)) if player_config else 1000
            if self.app_profile == APP_PROFILE_DEPLOY:
                budget_ms = min(budget_ms, DEPLOY_TIME_BUDGET_CAP_MS)
            think_timeout_s = max(2.0, budget_ms / 1000.0 + 1.0)
            stats = {
                "timeBudgetMs": budget_ms,
                "timeSpentMs": 0,
                "nodesEvaluated": 0,
                "maxDepthReached": 0,
            }
            try:
                def run_think():
                    engine_url = os.getenv("ENGINE_URL")
                    if engine_url:
                        body = json.dumps({
                            "gameState": {
                                "board": game.board.grid.tolist(),
                                "pieces_used": {pl.name: list(game.board.player_pieces_used[pl]) for pl in EnginePlayer},
                                "current_player": player.name,
                                "move_count": int(game.get_move_count()),
                            },
                            "legalMoves": [self._convert_move_back(m).dict() for m in legal_moves],
                            "timeBudgetMs": budget_ms,
                        }).encode("utf-8")
                        req = urllib.request.Request(
                            f"{engine_url.rstrip('/')}/think",
                            data=body,
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        try:
                            with urllib.request.urlopen(req, timeout=max(2.0, budget_ms / 1000.0 + 1.0)) as resp:
                                payload = json.loads(resp.read().decode("utf-8"))
                                mv = payload.get("move")
                                move_obj = EngineMove(**mv) if mv else None
                                return move_obj, payload.get("stats", {})
                        except Exception as engine_exc:
                            logger.warning(f"Engine service failed, falling back local MCTS: {engine_exc}")
                    if self.app_profile == APP_PROFILE_DEPLOY and is_gameplay_adapter(agent):
                        move_obj, adapter_stats = agent.choose_move(game.board, player, legal_moves, budget_ms)
                        return move_obj, adapter_stats
                    think_fn = getattr(agent, 'think', None)
                    if callable(think_fn) and 'think' in getattr(agent, '__dict__', {}):
                        result = think_fn(game.board, player, legal_moves, budget_ms)
                        return result.get('move'), result.get('stats', {})
                    move_result = agent.select_action(game.board, player, legal_moves)
                    return move_result, {}

                move, think_stats = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, run_think),
                    timeout=think_timeout_s
                )
                if isinstance(think_stats, dict):
                    stats.update(think_stats)
                stats['timeBudgetMs'] = budget_ms
                stats['timeSpentMs'] = int((time.perf_counter() - start_agent) * 1000)
                end_agent = time.perf_counter()
                agent_time = end_agent - start_agent
                logger.info(f"AGENT MOVE agent_selection: move={move} selected in {agent_time:.4f}s")
            except asyncio.TimeoutError:
                logger.warning(f"AGENT MOVE: Agent {type(agent).__name__} timed out after {think_timeout_s:.2f}s in game {game_id} for player {player.name}")
                self._record_pass(game_id, player, agent_type=type(agent).__name__, reason="timeout")
                game.board._update_current_player()
                game._check_game_over()
                return
            except Exception as e:
                logger.error(f"AGENT MOVE: Agent {type(agent).__name__} raised exception in game {game_id} for player {player.name}: {e}")
                import traceback
                traceback.print_exc()
                self._record_pass(game_id, player, agent_type=type(agent).__name__, reason="error")
                game.board._update_current_player()
                game._check_game_over()
                return
            
            # Make the move (only reached if agent successfully returned a move)
            state_before = game.get_board_copy()
            turn_index = game.get_move_count()
            start_apply = time.perf_counter()
            success = game.make_move(move, player)
            end_apply = time.perf_counter()
            apply_time = end_apply - start_apply
            logger.info(f"AGENT MOVE apply: success={success} in {apply_time:.4f}s")
            
            if success:
                self._log_step(game_id, turn_index, player.value, state_before, move, game.board)
                # Log the move details
                player_name = self._convert_player_back(player).value
                logger.info(f"Player {player_name} placed a piece at {move.anchor_row},{move.anchor_col} (piece_id={move.piece_id}, orientation={move.orientation})")
                # Store MCTS explainability for next game state fetch
                game_data['last_mcts_top_moves'] = stats.get('topMoves') or []
                # Prepare move information for broadcast
                last_move = {
                    "piece_id": move.piece_id,
                    "orientation": move.orientation,
                    "anchor_row": move.anchor_row,
                    "anchor_col": move.anchor_col,
                    "player": player_name,
                    "stats": stats,
                }
                seq = game_data.get("_event_sequence", 0)
                game_data["_event_sequence"] = seq + 1
                game_data['move_records'].append({
                    "sequenceIndex": seq,
                    "moveIndex": game.get_move_count(),
                    "player": player_name,
                    "agentType": type(agent).__name__,
                    "isHuman": False,
                    "move": last_move,
                    "stats": stats,
                })
                game_data['last_turn_started_at'] = time.perf_counter()
                game_data['last_turn_player'] = game.get_current_player()
                
                end_total = time.perf_counter()
                total_time = end_total - start_total
                logger.info(f"AGENT MOVE timing: total={total_time:.4f}s, legal={legal_time:.4f}s, agent={agent_time:.4f}s, apply={apply_time:.4f}s, broadcast={broadcast_time:.4f}s")
            else:
                logger.warning(f"AGENT MOVE: Invalid move from agent {type(agent).__name__} for player {player.name}")
                
        except Exception as e:
            logger.error(f"AGENT MOVE ERROR: {str(e)} for player {player.name}")
            import traceback
            traceback.print_exc()
    
    def _record_pass(self, game_id: str, player: EnginePlayer, agent_type: str = "human", reason: str = "no_moves"):
        """Record a pass (skip turn) in move_records for replay."""
        if game_id not in self.games:
            return
        game_data = self.games[game_id]
        player_name = self._convert_player_back(player).value
        seq = game_data.get("_event_sequence", 0)
        game_data["_event_sequence"] = seq + 1
        game_data['move_records'].append({
            "sequenceIndex": seq,
            "moveIndex": None,
            "player": player_name,
            "agentType": agent_type,
            "isHuman": agent_type == "human",
            "isPass": True,
            "reason": reason,
            "move": None,
            "stats": {},
        })

    def _init_strategy_logger(self, game_id: str) -> None:
        """Create StrategyLogger for game if enabled. Call on_reset."""
        if not is_strategy_logger_enabled():
            return
        try:
            base_dir = get_strategy_log_dir()
            log_dir = get_strategy_log_dir_for_game(base_dir, game_id)
            from analytics.logging.logger import StrategyLogger
            sl = StrategyLogger(log_dir=log_dir)
            self._strategy_loggers[game_id] = sl
            game_data = self.games[game_id]
            game = game_data["game"]
            config = game_data["config"]
            agent_ids = {}
            for pc in config.players:
                ep = self._convert_player(pc.player)
                name = pc.agent_type.value if hasattr(pc.agent_type, "value") else str(pc.agent_type)
                if pc.agent_config and "difficulty" in (pc.agent_config or {}):
                    name = f"{name}_{pc.agent_config['difficulty']}"
                agent_ids[ep.value] = name
            sl.on_reset(game_id, seed=0, agent_ids=agent_ids, config={})
            logger.info(f"StrategyLogger initialized for game {game_id}")
        except Exception as e:
            logger.warning(f"StrategyLogger init failed for {game_id}: {e}")

    def _get_strategy_logger(self, game_id: str):
        """Return StrategyLogger for game_id or None."""
        return self._strategy_loggers.get(game_id)

    def _log_step(self, game_id: str, turn_index: int, player_id: int,
                  state_before, move: EngineMove, next_state) -> None:
        """Log step to StrategyLogger if enabled."""
        sl = self._get_strategy_logger(game_id)
        if not sl:
            return
        try:
            sl.on_step(game_id, turn_index, player_id, state_before, move, next_state)
        except Exception as e:
            logger.warning(f"StrategyLogger on_step failed for {game_id}: {e}")

    def _log_game_end(self, game_id: str) -> None:
        """Log game end to StrategyLogger if enabled."""
        sl = self._get_strategy_logger(game_id)
        if not sl:
            return
        try:
            game_data = self.games.get(game_id)
            if not game_data:
                return
            game = game_data["game"]
            scores = {p.value: int(game.get_score(p)) for p in EnginePlayer}
            winner = game.board.get_winner()
            winner_id = winner.value if winner else None
            num_turns = int(game.get_move_count())
            sl.on_game_end(game_id, scores, winner_id, num_turns)
            logger.info(f"StrategyLogger on_game_end for {game_id}")
        except Exception as e:
            logger.warning(f"StrategyLogger on_game_end failed for {game_id}: {e}")
        finally:
            self._strategy_loggers.pop(game_id, None)

    async def _end_game(self, game_id: str):
        """End the game and persist analysis payload."""
        if game_id not in self.games:
            return

        game_data = self.games[game_id]
        game = game_data['game']
        self._log_game_end(game_id)
        game_data['status'] = GameStatus.FINISHED
        game_data['updated_at'] = datetime.now()
        winner = game.board.get_winner()
        game_data['winner'] = self._convert_player_back(winner).value if winner else None

        try:
            db = get_database()
            game_doc = {
                "game_id": game_id,
                "created_at": game_data['created_at'],
                "finished_at": game_data['updated_at'],
                "winner": game_data['winner'],
                "move_count": int(game.get_move_count()),
                "scores": {player.name: int(game.get_score(player)) for player in EnginePlayer},
                "players": [p.dict() for p in game_data['config'].players],
            }
            await db.game_records.update_one({"game_id": game_id}, {"$set": game_doc}, upsert=True)
            await db.move_records.delete_many({"game_id": game_id})
            if game_data.get('move_records'):
                await db.move_records.insert_many([
                    {"game_id": game_id, **record} for record in game_data['move_records']
                ])
        except Exception as exc:
            logger.warning(f"Could not persist game analytics for {game_id}: {exc}")
    
    async def make_move(self, game_id: str, move_request: MoveRequest) -> MoveResponse:
        """Make a move in the game."""
        if game_id not in self.games:
            raise HTTPException(status_code=404, detail="Game not found")
        
        game_data = self.games[game_id]
        game = game_data['game']
        
        if game_data['status'] != GameStatus.IN_PROGRESS:
            return MoveResponse(
                success=False,
                message="Game is not in progress",
                game_state=self._get_game_state(game_id)
            )
        
        # Convert move - MoveRequest has move data nested under 'move' field
        move_data = move_request.move
        engine_move = EngineMove(move_data.piece_id, move_data.orientation, move_data.anchor_row, move_data.anchor_col)
        player = self._convert_player(move_request.player)
        
        state_before = game.get_board_copy()
        turn_index = game.get_move_count()
        # Make the move
        success = game.make_move(engine_move, player)
        
        if success:
            self._log_step(game_id, turn_index, player.value, state_before, engine_move, game.board)
            game_data['updated_at'] = datetime.now()
            return MoveResponse(
                success=True,
                message="Move made successfully",
                game_state=self._get_game_state(game_id)
            )
        else:
            return MoveResponse(
                success=False,
                message="Invalid move",
                game_state=self._get_game_state(game_id)
            )
    
    async def _process_human_pass(self, game_id: str, player: Player) -> MoveResponse:
        """
        Process a human player's pass (skip turn).
        
        Args:
            game_id: Game identifier
            player: Player requesting to pass
            
        Returns:
            MoveResponse indicating success or failure
        """
        try:
            logger.info(f"Processing human pass for {player.value}")
            
            # Get game data
            if game_id not in self.games:
                return MoveResponse(success=False, message="Game not found", game_over=False)
            
            game_data = self.games[game_id]
            game = game_data['game']
            
            # Convert schema player to engine player
            engine_player = self._convert_player(player)
            
            # Verify it's the player's turn
            current_player = game.get_current_player()
            if current_player != engine_player:
                return MoveResponse(
                    success=False, 
                    message=f"It's not {player.value}'s turn", 
                    game_over=False
                )
            
            # Optionally check if player has legal moves (log warning if they do)
            legal_moves = game.get_legal_moves(engine_player)
            if legal_moves:
                logger.warning(f"Player {player.value} is passing but has {len(legal_moves)} legal moves available")
            
            # Record pass for replay
            self._record_pass(game_id, engine_player, agent_type="human", reason="no_moves")

            # Advance to next player (same logic as agent skip)
            game.board._update_current_player()
            logger.info(f"Turn advanced to {game.get_current_player().name}")
            
            # Update game data
            game_data['updated_at'] = datetime.now()
            
            # Re-check game-over (no piece was placed, so make_move never ran _check_game_over)
            game._check_game_over()
            
            # Check if game is over
            game_over = game.is_game_over()
            if game_over:
                await self._end_game(game_id)
            
            return MoveResponse(
                success=True,
                message="Turn passed successfully",
                game_state=self._get_game_state(game_id)
            )
                
        except Exception as e:
            logger.error(f"Error in pass processing: {e}")
            return MoveResponse(
                success=False,
                message=f"Error passing turn: {str(e)}",
                game_over=False
            )
    
    async def _process_human_move_immediately(self, game_id: str, move_request: MoveRequest) -> MoveResponse:
        """Process human move immediately without waiting for turn loop."""
        start_total = time.perf_counter()
        try:
            # Log incoming move with raw data
            move_data = move_request.move
            logger.info(f"HUMAN MOVE START: game_id={game_id}, player={move_request.player}, raw_move={move_data.dict() if hasattr(move_data, 'dict') else {'piece_id': move_data.piece_id, 'orientation': move_data.orientation, 'anchor_row': move_data.anchor_row, 'anchor_col': move_data.anchor_col}}")
            
            # Get game data
            if game_id not in self.games:
                return MoveResponse(success=False, message="Game not found", game_over=False)
            
            game_data = self.games[game_id]
            game = game_data['game']
            
            # Convert schema player to engine player
            engine_player = self._convert_player(move_request.player)
            
            # Pre-check 1: Verify it's the player's turn
            current_player = game.get_current_player()
            if current_player != engine_player:
                logger.warning(f"HUMAN MOVE: Not player's turn. Current={current_player.name}, Requested={engine_player.name}")
                return MoveResponse(
                    success=False, 
                    message="It is not your turn.", 
                    game_over=False
                )
            
            # Create move object
            piece_id = move_data.piece_id
            orientation = move_data.orientation
            anchor_row = move_data.anchor_row
            anchor_col = move_data.anchor_col
            
            # Pre-check 2: Piece already used
            if piece_id in game.board.player_pieces_used[engine_player]:
                logger.warning(f"HUMAN MOVE: Piece {piece_id} already used by {engine_player.name}")
                return MoveResponse(
                    success=False,
                    message="This piece has already been used.",
                    game_state=self._get_game_state(game_id)
                )
            
            # Pre-check 3: Invalid orientation index
            if piece_id not in game.move_generator.piece_orientations_cache:
                logger.warning(f"HUMAN MOVE: Invalid piece_id {piece_id}")
                return MoveResponse(
                    success=False,
                    message="Invalid piece ID.",
                    game_state=self._get_game_state(game_id)
                )
            
            orientations = game.move_generator.piece_orientations_cache[piece_id]
            if orientation < 0 or orientation >= len(orientations):
                logger.warning(f"HUMAN MOVE: Invalid orientation {orientation} for piece {piece_id} (max={len(orientations)-1})")
                return MoveResponse(
                    success=False,
                    message="Invalid orientation for this piece.",
                    game_state=self._get_game_state(game_id)
                )
            
            # Pre-check 4: Out of bounds
            from engine.pieces import PiecePlacement
            orientation_shape = orientations[orientation]
            if not PiecePlacement.can_place_piece_at(
                (game.board.SIZE, game.board.SIZE),
                orientation_shape,
                anchor_row,
                anchor_col
            ):
                logger.warning(f"HUMAN MOVE: Out of bounds - piece_id={piece_id}, anchor=({anchor_row},{anchor_col})")
                return MoveResponse(
                    success=False,
                    message="Move is out of bounds.",
                    game_state=self._get_game_state(game_id)
                )
            
            # Create engine move object
            engine_move = EngineMove(piece_id, orientation, anchor_row, anchor_col)
            
            state_before = game.get_board_copy()
            turn_index = game.get_move_count()
            # Make the move immediately
            start_make_move = time.perf_counter()
            success = game.make_move(engine_move, engine_player)
            end_make_move = time.perf_counter()
            make_move_time = end_make_move - start_make_move
            logger.info(f"HUMAN MOVE make_move: success={success} in {make_move_time:.4f}s")
            
            if success:
                self._log_step(game_id, turn_index, engine_player.value, state_before, engine_move, game.board)
                # Log the move details
                player_name = move_request.player.value if hasattr(move_request.player, 'value') else str(move_request.player)
                logger.info(f"Player {player_name} placed a piece at {anchor_row},{anchor_col} (piece_id={piece_id}, orientation={orientation})")
                
                # Prepare move information for broadcast
                last_move = {
                    "piece_id": piece_id,
                    "orientation": orientation,
                    "anchor_row": anchor_row,
                    "anchor_col": anchor_col,
                    "player": player_name
                }
                
                # Update game data
                game_data['updated_at'] = datetime.now()
                user_move_time_ms = int((time.perf_counter() - game_data.get('last_turn_started_at', time.perf_counter())) * 1000)
                seq = game_data.get("_event_sequence", 0)
                game_data["_event_sequence"] = seq + 1
                game_data['move_records'].append({
                    "sequenceIndex": seq,
                    "moveIndex": game.get_move_count(),
                    "player": player_name,
                    "agentType": "human",
                    "isHuman": True,
                    "move": last_move,
                    "stats": {
                        "userMoveTimeMs": max(user_move_time_ms, 0),
                    },
                })
                game_data['last_turn_started_at'] = time.perf_counter()
                game_data['last_turn_player'] = game.get_current_player()
                
                end_total = time.perf_counter()
                total_time = end_total - start_total
                logger.info(f"HUMAN MOVE timing: total={total_time:.4f}s, make_move={make_move_time:.4f}s, broadcast={broadcast_time:.4f}s")
                
                # Check if game is over
                game_over = game.is_game_over()
                winner = None
                if game_over:
                    winner_engine = game.board.get_winner()
                    if winner_engine:
                        winner = self._convert_player_back(winner_engine)
                    await self._end_game(game_id)
                
                return MoveResponse(
                    success=True,
                    message="Move made successfully",
                    game_state=self._get_game_state(game_id)
                )
            else:
                # Move failed after all pre-checks passed - must be a Blokus placement rule violation
                logger.warning(f"HUMAN MOVE: Move violates Blokus rules - piece_id={piece_id}, anchor=({anchor_row},{anchor_col})")
                return MoveResponse(
                    success=False,
                    message="Move violates Blokus placement rules (overlap, edge contact, or missing corner connection).",
                    game_state=self._get_game_state(game_id)
                )
                
        except Exception as e:
            logger.error(f"HUMAN MOVE ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return MoveResponse(
                success=False,
                message=f"Error making move: {str(e)}",
                game_over=False
            )
    
    async def force_finish_game(self, game_id: str) -> None:
        """Force-complete a game and persist analytics."""
        if game_id not in self.games:
            raise HTTPException(status_code=404, detail="Game not found")
        await self._end_game(game_id)

    def get_game_state(self, game_id: str) -> GameState:
        """Get the current game state."""
        if game_id not in self.games:
            raise HTTPException(status_code=404, detail="Game not found")
        
        return self._get_game_state(game_id)
    
    def _get_game_state(self, game_id: str) -> GameState:
        """Internal method to get game state."""
        game_data = self.games[game_id]
        game = game_data['game']
        
        # Convert board to list of lists
        board = game.board.grid.tolist()
        
        # Get scores
        scores = {}
        for player in EnginePlayer:
            scores[player.name] = game.get_score(player)
        
        # Get pieces used
        pieces_used = {}
        for player in EnginePlayer:
            pieces_used[player.name] = list(game.board.player_pieces_used[player])
        
        # Get legal moves (optimized: reuse move_generator from game)
        legal_moves = []
        engine_moves = []
        heatmap = [[0.0 for _ in range(20)] for _ in range(20)]  # Initialize empty heatmap
        if not game.is_game_over():
            engine_moves = game.get_legal_moves()
            legal_moves = [self._convert_move_back(move) for move in engine_moves]
            
            # Calculate heatmap: map legal_moves to 20x20 grid (1 = legal, 0 = illegal)
            # OPTIMIZED: Use cached move_generator and cached positions
            move_generator = game.move_generator
            for engine_move in engine_moves:
                # Use cached positions if available
                if engine_move.piece_id in move_generator.piece_position_cache:
                    cached_positions = move_generator.piece_position_cache[engine_move.piece_id][engine_move.orientation]
                    for rel_r, rel_c in cached_positions:
                        row = engine_move.anchor_row + rel_r
                        col = engine_move.anchor_col + rel_c
                        if 0 <= row < 20 and 0 <= col < 20:
                            heatmap[row][col] = 1.0
                else:
                    # Fallback to original method
                    orientations = move_generator.piece_orientations_cache[engine_move.piece_id]
                    orientation = orientations[engine_move.orientation]
                    for i in range(orientation.shape[0]):
                        for j in range(orientation.shape[1]):
                            if orientation[i, j] == 1:
                                row = engine_move.anchor_row + i
                                col = engine_move.anchor_col + j
                                if 0 <= row < 20 and 0 <= col < 20:
                                    heatmap[row][col] = 1.0

        # Mobility metrics per docs/metrics/mobility.md (canonical source)
        current_player_engine = game.get_current_player()
        pieces_used_current = list(game.board.player_pieces_used[current_player_engine])
        mobility = compute_player_mobility_metrics(engine_moves, pieces_used_current)
        
        center_control = game._calculate_center_bonus(current_player_engine) // 2
        frontier_size = len(game.board.get_frontier(current_player_engine))
        
        mobility_metrics = {
            "totalPlacements": mobility.totalPlacements,
            "totalOrientationNormalized": mobility.totalOrientationNormalized,
            "totalCellWeighted": mobility.totalCellWeighted,
            "buckets": mobility.buckets,
            "centerControl": center_control,
            "frontierSize": frontier_size,
        }
        mcts_top_moves = game_data.get('last_mcts_top_moves')

        return GameState(
            game_id=game_id,
            status=game_data['status'],
            current_player=self._convert_player_back(game.get_current_player()),
            board=board,
            scores=scores,
            pieces_used=pieces_used,
            move_count=game.get_move_count(),
            game_over=game.is_game_over(),
            winner=self._convert_player_back(game.winner) if game.winner else None,
            legal_moves=legal_moves,
            created_at=game_data['created_at'],
            updated_at=game_data['updated_at'],
            players=[p.dict() for p in game_data['config'].players],
            heatmap=heatmap,
            mobility_metrics=mobility_metrics,
            mcts_top_moves=mcts_top_moves
        )
    
    def get_available_agents(self) -> List[AgentInfo]:
        """Get list of available agents."""
        return [
            AgentInfo(
                type=AgentType.RANDOM,
                name="Random Agent",
                description="Makes random legal moves"
            ),
            AgentInfo(
                type=AgentType.HEURISTIC,
                name="Heuristic Agent",
                description="Uses heuristic evaluation for move selection"
            ),
            AgentInfo(
                type=AgentType.MCTS,
                name="MCTS Agent",
                description="Uses Monte Carlo Tree Search"
            ),
            AgentInfo(
                type=AgentType.HUMAN,
                name="Human Player",
                description="Human player controlled via WebSocket"
            )
        ]
    
    def _convert_player(self, player: Player) -> EnginePlayer:
        """Convert schema Player to engine Player."""
        return EnginePlayer[player.value]
    
    def _convert_player_back(self, player: EnginePlayer) -> Player:
        """Convert engine Player to schema Player."""
        return Player(player.name)
    
    def _convert_move(self, move: Move) -> EngineMove:
        """Convert schema Move to engine Move."""
        return EngineMove(move.piece_id, move.orientation, move.anchor_row, move.anchor_col)
    
    def _convert_move_back(self, move: EngineMove) -> Move:
        """Convert engine Move to schema Move."""
        return Move(
            piece_id=move.piece_id,
            orientation=move.orientation,
            anchor_row=move.anchor_row,
            anchor_col=move.anchor_col
        )


# Global game manager instance (rebuilt per app profile in create_app)
APP_PROFILE = get_app_profile()
_current_app_profile = APP_PROFILE
game_manager = GameManager(app_profile=APP_PROFILE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    In research profile we connect MongoDB (best effort).
    In deploy profile we skip MongoDB startup.
    """
    logger.info("Starting Blokus RL Web API...")
    if _current_app_profile == APP_PROFILE_RESEARCH:
        try:
            await connect_to_mongo()
            logger.info("MongoDB connection established")
            try:
                database = get_database()
                await database.command("ping")
                logger.info("✅ MongoDB connection validated successfully")
            except Exception as e:
                logger.error(f"❌ MongoDB connection validation failed: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB during startup: {e}")
            logger.warning("Server will start without MongoDB connection. Some features may be unavailable.")
    else:
        logger.info("Deploy profile: skipping MongoDB startup.")

    yield

    logger.info("Shutting down Blokus RL Web API...")
    if _current_app_profile == APP_PROFILE_RESEARCH:
        await close_mongo_connection()


def health():
    """Lightweight health endpoint (deploy and research)."""
    return {"ok": True, "profile": _current_app_profile}


async def create_game(config: GameConfig):
    """Create a new game."""
    try:
        if _current_app_profile == APP_PROFILE_DEPLOY:
            config = normalize_deploy_game_config(config)

        game_id = game_manager.create_game(config)
        game_state = game_manager.get_game_state(game_id)

        return GameCreateResponse(
            game_id=game_id,
            game_state=game_state,
            message="Game created successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def root():
    """Root endpoint with API information."""
    if _current_app_profile == APP_PROFILE_RESEARCH:
        return {
            "message": "Blokus RL Web API",
            "version": "1.0.0",
            "docs": "/docs",
            "endpoints": {
                "agents": "/api/agents",
                "games": "/api/games",
                "websocket": "/ws/games/{game_id}",
                "health": "/api/health/db"
            }
        }
    return {
        "message": "Blokus RL Web API",
        "version": "1.0.0",
        "docs": "/docs",
        "profile": _current_app_profile,
        "endpoints": {
            "agents": "/api/agents",
            "games": "/api/games",
            "websocket": "/ws/games/{game_id}",
            "health": "/health",
        },
    }


async def health_check_db():
    """
    Database health check endpoint.
    
    Tests MongoDB connectivity by performing a simple ping operation.
    Returns connection status without exposing sensitive information.
    """
    try:
        db = get_database()
        # Perform a simple ping to test connectivity
        await db.command("ping")
        return {
            "ok": True,
            "db": "connected",
            "database": db.name
        }
    except RuntimeError as e:
        # Database not initialized
        logger.error(f"Database health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": "Database not initialized",
                "message": "MongoDB connection was not established during startup"
            }
        )
    except Exception as e:
        # Connection error
        logger.error(f"Database health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": "Database connection failed",
                "message": "Unable to connect to MongoDB"
            }
        )


async def mongo_debug():
    """
    Debug endpoint to verify MongoDB connectivity and list collections.
    
    This endpoint helps confirm:
    - MongoDB credentials are correct
    - The db object is accessible and functional
    - Available collections in the database
    """
    try:
        database = get_database()
        collections = await database.list_collection_names()
        return {"ok": True, "collections": collections, "database": database.name}
    except RuntimeError as e:
        return {"ok": False, "error": f"MongoDB not initialized: {str(e)}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def get_game(game_id: str):
    """Get game state."""
    return game_manager.get_game_state(game_id)


async def make_move(game_id: str, move_request: MoveRequest):
    """Make a move in the game."""
    return await game_manager.make_move(game_id, move_request)


async def advance_turn(game_id: str):
    """Advance the turn for the connected agent based on polling frontend."""
    return await game_manager.advance_turn(game_id)


from schemas.game_state import Player
from pydantic import BaseModel

class PassRequest(BaseModel):
    player: str

async def pass_turn(game_id: str, pass_request: PassRequest):
    """Process a pass turn for a human player."""
    player = Player(pass_request.player.upper())
    return await game_manager._process_human_pass(game_id, player)


async def finish_game(game_id: str):
    """Force-finish a game (for smoke tests/admin tooling)."""
    await game_manager.force_finish_game(game_id)
    return {"ok": True, "game_id": game_id}


async def get_agents():
    """Get list of available agents."""
    agents = game_manager.get_available_agents()
    if _current_app_profile == APP_PROFILE_DEPLOY:
        return [
            agent for agent in agents
            if agent.type in (AgentType.MCTS, AgentType.HUMAN)
        ]
    return agents


async def list_games():
    """List all active games."""
    games = []
    for game_id in game_manager.games:
        games.append(game_manager.get_game_state(game_id))
    return games


async def _replay_to_index(game_id: str, move_index: int) -> tuple:
    """
    Replay a game up to move_index (0-based event index) and return (game, game_doc, move_docs).
    move_index=0 returns initial board; move_index=1 returns state after first event, etc.
    """
    game_data = game_manager.games.get(game_id)
    if game_data:
        game_doc = {
            "game_id": game_id,
            "created_at": game_data["created_at"],
            "finished_at": game_data.get("updated_at"),
            "winner": game_data.get("winner"),
            "move_count": int(game_data["game"].get_move_count()),
            "scores": {p.name: game_data["game"].get_score(p) for p in EnginePlayer},
            "players": [p.dict() for p in game_data["config"].players],
        }
        move_docs = list(game_data.get("move_records", []))
        move_docs.sort(key=lambda m: m.get("sequenceIndex", m.get("moveIndex", 0) if m.get("moveIndex") is not None else 9999))
    else:
        try:
            db = get_database()
            game_doc = await db.game_records.find_one({"game_id": game_id})
            if not game_doc:
                return None, None, None
            move_docs = await db.move_records.find({"game_id": game_id}).to_list(length=2000)
            move_docs.sort(key=lambda m: m.get("sequenceIndex", m.get("moveIndex", 0) if m.get("moveIndex") is not None else 9999))
        except Exception:
            return None, None, None

    from engine.game import BlokusGame
    from engine.board import Player as EnginePlayerEnum

    replay_game = BlokusGame()
    player_map = {"RED": EnginePlayerEnum.RED, "BLUE": EnginePlayerEnum.BLUE, "YELLOW": EnginePlayerEnum.YELLOW, "GREEN": EnginePlayerEnum.GREEN}

    for i, rec in enumerate(move_docs):
        if i >= move_index:
            break
        if rec.get("isPass"):
            replay_game.board._update_current_player()
        else:
            mv = rec.get("move")
            if mv:
                engine_move = EngineMove(mv["piece_id"], mv["orientation"], mv["anchor_row"], mv["anchor_col"])
                pl = player_map.get(rec.get("player", ""), EnginePlayerEnum.RED)
                replay_game.make_move(engine_move, pl)

    return replay_game, game_doc, move_docs


async def get_game_replay(game_id: str, move_index: int = -1):
    """
    Return board state at a specific move index for step-by-step replay.
    move_index: 0-based event index (0=initial, 1=after first move/pass, etc). -1 = final state.
    """
    result = await _replay_to_index(game_id, move_index)
    if result[0] is None:
        raise HTTPException(status_code=404, detail="Game not found")
    replay_game, game_doc, move_docs = result
    if move_index < 0:
        move_index = len(move_docs)
    scores = {p.name: replay_game.get_score(p) for p in EnginePlayer}
    return {
        "game_id": game_id,
        "move_index": move_index,
        "total_events": len(move_docs),
        "board": replay_game.board.grid.tolist(),
        "current_player": replay_game.board.current_player.name,
        "scores": scores,
        "pieces_used": {p.name: list(replay_game.board.player_pieces_used[p]) for p in EnginePlayer},
        "move_count": replay_game.board.move_count,
        "game_over": replay_game.board.game_over,
        "event": move_docs[move_index - 1] if 0 < move_index <= len(move_docs) else None,
    }


async def get_analysis_steps(game_id: str, limit: int = 100, offset: int = 0):
    """
    Return StepLog entries for game_id from StrategyLogger output.
    Paginated in chronological order.
    """
    from webapi.strategy_logger_config import get_strategy_log_dir
    from analytics.logging.reader import load_steps_for_game

    base_dir = get_strategy_log_dir()
    steps = load_steps_for_game(base_dir, game_id)
    if not steps:
        return {"game_id": game_id, "steps": [], "total": 0, "limit": limit, "offset": offset}

    total = len(steps)
    page = steps[offset : offset + limit]
    return {"game_id": game_id, "steps": page, "total": total, "limit": limit, "offset": offset}


async def get_analysis_summary(game_id: str):
    """
    Return aggregates for UI charts: mobility curves, deltas over time.
    Derived from StrategyLogger steps.jsonl.
    """
    from webapi.strategy_logger_config import get_strategy_log_dir
    from analytics.logging.reader import load_steps_for_game

    base_dir = get_strategy_log_dir()
    steps = load_steps_for_game(base_dir, game_id)
    if not steps:
        return {"game_id": game_id, "mobilityCurve": [], "deltas": [], "totalSteps": 0}

    mobility_curve = []
    deltas = []
    for s in steps:
        turn = s.get("turn_index", 0)
        pid = s.get("player_id")
        before = s.get("legal_moves_before", 0)
        after = s.get("legal_moves_after", 0)
        metrics = s.get("metrics") or {}
        delta = metrics.get("mobility_me_delta", after - before)
        mobility_curve.append({"turn_index": turn, "player_id": pid, "legal_moves_before": before, "legal_moves_after": after})
        deltas.append({"turn_index": turn, "player_id": pid, "delta": delta})

    return {
        "game_id": game_id,
        "mobilityCurve": mobility_curve,
        "deltas": deltas,
        "totalSteps": len(steps),
    }


async def get_game_analysis(game_id: str):
    """Return per-move AI/user metrics and aggregate game analysis."""
    try:
        db = get_database()
        game_doc = await db.game_records.find_one({"game_id": game_id})
        move_docs = await db.move_records.find({"game_id": game_id}).to_list(length=2000)
        move_docs.sort(key=lambda m: m.get("sequenceIndex", m.get("moveIndex", 0) if m.get("moveIndex") is not None else 9999))
    except Exception:
        game_doc = None
        move_docs = []

    if not game_doc:
        game_data = game_manager.games.get(game_id)
        if not game_data:
            raise HTTPException(status_code=404, detail="Game not found")
        game_doc = {
            "game_id": game_id,
            "created_at": game_data["created_at"],
            "finished_at": game_data["updated_at"],
            "winner": game_data.get("winner"),
            "move_count": int(game_data["game"].get_move_count()),
            "scores": {player.name: game_data["game"].get_score(player) for player in EnginePlayer},
        }
        move_docs = list(game_data.get("move_records", []))
        move_docs.sort(key=lambda m: m.get("sequenceIndex", m.get("moveIndex", 0) if m.get("moveIndex") is not None else 9999))

    ai_moves = [m for m in move_docs if not m.get("isHuman")]
    user_moves = [m for m in move_docs if m.get("isHuman")]
    total_ai_nodes = sum((m.get("stats") or {}).get("nodesEvaluated", 0) for m in ai_moves)
    total_ai_think = sum((m.get("stats") or {}).get("timeSpentMs", 0) for m in ai_moves)
    max_depth = max([((m.get("stats") or {}).get("maxDepthReached", 0)) for m in ai_moves] + [0])
    game_duration_ms = int((game_doc["finished_at"] - game_doc["created_at"]).total_seconds() * 1000) if game_doc.get("finished_at") and game_doc.get("created_at") else 0
    avg_user_move_ms = int(sum((m.get("stats") or {}).get("userMoveTimeMs", 0) for m in user_moves) / max(len(user_moves), 1))

    return {
        "game": game_doc,
        "moves": move_docs,
        "aggregates": {
            "totalAiThinkTimeMs": total_ai_think,
            "totalAiNodesEvaluated": total_ai_nodes,
            "nodesPerSecond": round((total_ai_nodes / max(total_ai_think, 1)) * 1000, 2),
            "maxDepthReached": max_depth,
            "gameDurationMs": game_duration_ms,
            "avgUserMoveTimeMs": avg_user_move_ms,
            "userMoveCount": len(user_moves),
        },
    }
async def get_history(limit: int = 20):
    """List recently finished games with summary metrics."""
    records = []
    try:
        db = get_database()
        docs = await db.game_records.find({}).sort("finished_at", -1).limit(limit).to_list(length=limit)
        move_docs = await db.move_records.find({}).to_list(length=200000)
        move_map = {}
        for move in move_docs:
            gid = move.get("game_id")
            move_map.setdefault(gid, []).append(move)
        for doc in docs:
            gid = doc.get("game_id")
            moves = move_map.get(gid, [])
            ai_moves = [m for m in moves if not m.get("isHuman")]
            user_moves = [m for m in moves if m.get("isHuman")]
            total_ai_nodes = sum((m.get("stats") or {}).get("nodesEvaluated", 0) for m in ai_moves)
            total_ai_think = sum((m.get("stats") or {}).get("timeSpentMs", 0) for m in ai_moves)
            records.append({
                "game_id": gid,
                "winner": doc.get("winner"),
                "move_count": doc.get("move_count", 0),
                "finished_at": doc.get("finished_at"),
                "gameDurationMs": int((doc["finished_at"] - doc["created_at"]).total_seconds() * 1000) if doc.get("created_at") and doc.get("finished_at") else 0,
                "avgUserMoveTimeMs": int(sum((m.get("stats") or {}).get("userMoveTimeMs", 0) for m in user_moves) / max(len(user_moves), 1)),
                "totalAiThinkTimeMs": total_ai_think,
                "totalAiNodesEvaluated": total_ai_nodes,
            })
        if records:
            return {"games": records}
    except Exception:
        pass

    finished = []
    for gid, g in game_manager.games.items():
        if g.get('status') != GameStatus.FINISHED:
            continue
        moves = g.get('move_records', [])
        ai_moves = [m for m in moves if not m.get("isHuman")]
        user_moves = [m for m in moves if m.get("isHuman")]
        total_ai_nodes = sum((m.get("stats") or {}).get("nodesEvaluated", 0) for m in ai_moves)
        total_ai_think = sum((m.get("stats") or {}).get("timeSpentMs", 0) for m in ai_moves)
        finished.append({
            "game_id": gid,
            "winner": g.get("winner"),
            "move_count": int(g['game'].get_move_count()),
            "finished_at": g.get('updated_at'),
            "gameDurationMs": int((g['updated_at'] - g['created_at']).total_seconds() * 1000),
            "avgUserMoveTimeMs": int(sum((m.get("stats") or {}).get("userMoveTimeMs", 0) for m in user_moves) / max(len(user_moves), 1)),
            "totalAiThinkTimeMs": total_ai_think,
            "totalAiNodesEvaluated": total_ai_nodes,
        })
    finished.sort(key=lambda x: x.get('finished_at') or datetime.min, reverse=True)
    return {"games": finished[:limit]}

async def get_trends():
    """Cross-game aggregate trends for analysis dashboard."""
    try:
        db = get_database()
        games = await db.game_records.find({}).to_list(length=5000)
        moves = await db.move_records.find({}).to_list(length=200000)
    except Exception:
        games = []
        moves = []

    if not games:
        for gid, g in game_manager.games.items():
            if g.get('status') != GameStatus.FINISHED:
                continue
            games.append({
                "game_id": gid,
                "winner": g.get("winner"),
                "created_at": g.get("created_at"),
                "finished_at": g.get("updated_at"),
            })
            for m in g.get("move_records", []):
                moves.append({"game_id": gid, **m})

    if not games:
        return {
            "gamesPlayed": 0,
            "winRate": {},
            "avgAiNodesByBudget": {},
            "avgUserMoveTimeMs": 0,
            "avgGameDurationMs": 0,
        }

    win_counts = {}
    durations = []
    for g in games:
        winner = g.get("winner") or "NONE"
        win_counts[winner] = win_counts.get(winner, 0) + 1
        if g.get("created_at") and g.get("finished_at"):
            durations.append((g["finished_at"] - g["created_at"]).total_seconds() * 1000)

    ai_moves = [m for m in moves if not m.get("isHuman")]
    user_moves = [m for m in moves if m.get("isHuman")]
    by_budget = {}
    for m in ai_moves:
        stats = m.get("stats") or {}
        budget = str(stats.get("timeBudgetMs", 0))
        by_budget.setdefault(budget, []).append(stats.get("nodesEvaluated", 0))

    return {
        "gamesPlayed": len(games),
        "winRate": {k: round(v / len(games), 3) for k, v in win_counts.items()},
        "avgAiNodesByBudget": {k: round(sum(v) / max(len(v), 1), 2) for k, v in by_budget.items()},
        "avgUserMoveTimeMs": round(sum((m.get("stats") or {}).get("userMoveTimeMs", 0) for m in user_moves) / max(len(user_moves), 1), 2),
        "avgGameDurationMs": round(sum(durations) / max(len(durations), 1), 2),
    }


# Training Runs API Endpoints

async def list_training_runs(
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
):
    """
    List training runs with optional filtering.
    
    Args:
        agent_id: Filter by agent ID
        status: Filter by status (running, completed, stopped, failed)
        limit: Maximum number of runs to return (default: 50)
        
    Returns:
        List of training runs sorted by start_time descending
    """
    try:
        db = get_database()
        collection = db.training_runs
        
        # Build query
        query = {}
        if agent_id:
            query["agent_id"] = agent_id
        if status:
            query["status"] = status
        
        # Fetch runs sorted by start_time descending
        cursor = collection.find(query).sort("start_time", -1).limit(limit)
        runs = await cursor.to_list(length=limit)
        
        # Convert ObjectId to string for JSON serialization
        for run in runs:
            if "_id" in run:
                run["id"] = str(run["_id"])
                del run["_id"]
            # Convert datetime to ISO format strings
            if "start_time" in run and isinstance(run["start_time"], datetime):
                run["start_time"] = run["start_time"].isoformat()
            if "end_time" in run and isinstance(run["end_time"], datetime):
                run["end_time"] = run["end_time"].isoformat()
        
        return runs
    except RuntimeError as e:
        logger.error(f"Database not available: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database not available"
        )
    except Exception as e:
        logger.error(f"Error listing training runs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing training runs: {str(e)}"
        )


async def get_training_run(run_id: str):
    """
    Get details of a specific training run.
    
    Args:
        run_id: Training run ID (UUID)
        
    Returns:
        Training run details including metrics
    """
    try:
        db = get_database()
        collection = db.training_runs
        
        run = await collection.find_one({"run_id": run_id})
        
        if not run:
            raise HTTPException(
                status_code=404,
                detail=f"Training run {run_id} not found"
            )
        
        # Convert ObjectId to string
        if "_id" in run:
            run["id"] = str(run["_id"])
            del run["_id"]
        
        # Convert datetime to ISO format strings
        if "start_time" in run and isinstance(run["start_time"], datetime):
            run["start_time"] = run["start_time"].isoformat()
        if "end_time" in run and isinstance(run["end_time"], datetime):
            run["end_time"] = run["end_time"].isoformat()
        
        return run
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"Database not available: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database not available"
        )
    except Exception as e:
        logger.error(f"Error getting training run: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting training run: {str(e)}"
        )


async def list_agents():
    """
    Get list of distinct agent IDs from training runs.
    
    Returns:
        List of unique agent IDs
    """
    try:
        db = get_database()
        collection = db.training_runs
        
        # Get distinct agent IDs
        agent_ids = await collection.distinct("agent_id")
        
        return {"agent_ids": agent_ids}
    except RuntimeError as e:
        logger.error(f"Database not available: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database not available"
        )
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing agents: {str(e)}"
        )


async def get_training_run_evaluations(run_id: str):
    """
    Get evaluation results for a specific training run.
    
    Args:
        run_id: Training run ID
        
    Returns:
        List of evaluation runs for this training run
    """
    try:
        db = get_database()
        collection = db.evaluation_runs
        
        evaluations = await collection.find({"training_run_id": run_id}).sort("created_at", -1).to_list(length=100)
        
        # Convert ObjectId to string and datetime to ISO format
        for eval_run in evaluations:
            if "_id" in eval_run:
                eval_run["id"] = str(eval_run["_id"])
                del eval_run["_id"]
            if "created_at" in eval_run and isinstance(eval_run["created_at"], datetime):
                eval_run["created_at"] = eval_run["created_at"].isoformat()
        
        return evaluations
    except RuntimeError as e:
        logger.error(f"Database not available: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database not available"
        )
    except Exception as e:
        logger.error(f"Error getting evaluations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting evaluations: {str(e)}"
        )


# Error handlers

async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP Error",
            message=exc.detail
        ).dict()
    )
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred"
        ).dict()
    )


def _normalize_profile(profile: Optional[str]) -> str:
    if profile in (APP_PROFILE_RESEARCH, APP_PROFILE_DEPLOY):
        return profile  # type: ignore[return-value]
    return get_app_profile()


def create_app(
    *,
    profile: Optional[str] = None,
    include_research_routes: Optional[bool] = None,
) -> FastAPI:
    global _current_app_profile

    _current_app_profile = _normalize_profile(profile)
    # Preserve object identity so existing imports (tests/tools) remain valid.
    game_manager.app_profile = _current_app_profile
    game_manager.games.clear()
    game_manager.agent_instances.clear()

    app_instance = FastAPI(
        title="Blokus RL Web API",
        description="Web API for Blokus RL game with REST and WebSocket support",
        version="1.0.0",
        lifespan=lifespan,
    )

    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_gameplay_routes(
        app_instance,
        health=health,
        root=root,
        create_game=create_game,
        get_game=get_game,
        make_move=make_move,
        get_agents=get_agents,
        list_games=list_games,
        advance_turn=advance_turn,
        pass_turn=pass_turn,
    )

    register_research = (
        include_research_routes
        if include_research_routes is not None
        else _current_app_profile == APP_PROFILE_RESEARCH
    )
    if register_research:
        register_research_routes(
            app_instance,
            health_check_db=health_check_db,
            mongo_debug=mongo_debug,
            get_game_analysis=get_game_analysis,
            get_game_replay=get_game_replay,
            get_analysis_steps=get_analysis_steps,
            get_analysis_summary=get_analysis_summary,
            get_history=get_history,
            get_trends=get_trends,
            list_training_runs=list_training_runs,
            get_training_run=get_training_run,
            list_agents=list_agents,
            get_training_run_evaluations=get_training_run_evaluations,
        )

    app_instance.add_exception_handler(HTTPException, http_exception_handler)
    app_instance.add_exception_handler(Exception, general_exception_handler)
    return app_instance


# Canonical app entrypoint (research by default unless APP_PROFILE overrides)
app = create_app(profile=APP_PROFILE)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
