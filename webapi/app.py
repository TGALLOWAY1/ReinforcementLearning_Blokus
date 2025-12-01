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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

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
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from mcts.mcts_agent import MCTSAgent
from agents.fast_mcts_agent import FastMCTSAgent
from schemas.game_state import (
    GameConfig, GameState, GameStatus, Player, AgentType, Move, StateUpdate,
    MoveRequest, MoveResponse, GameCreateResponse, AgentInfo, ErrorResponse
)
# MongoDB connection module (webapi/db/mongo.py)
from webapi.db.mongo import connect_to_mongo, close_mongo_connection, get_database
from webapi.db.models import TrainingRun, EvaluationRun


class GameManager:
    """Manages active games and their state."""
    
    def __init__(self):
        self.games: Dict[str, Dict[str, Any]] = {}
        self.websocket_connections: Dict[str, List[WebSocket]] = {}
        self.agent_instances: Dict[str, Any] = {}
    
    def create_game(self, config: GameConfig) -> str:
        """Create a new game."""
        game_id = config.game_id or str(uuid.uuid4())
        
        if game_id in self.games:
            raise HTTPException(status_code=400, detail="Game ID already exists")
        
        # Create game instance
        game = BlokusGame()
        
        # Store game data
        self.games[game_id] = {
            'game': game,
            'config': config,
            'status': GameStatus.WAITING,
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'websocket_connections': []
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
            
            if agent_type == AgentType.RANDOM:
                agents[player] = RandomAgent()
            elif agent_type == AgentType.HEURISTIC:
                agents[player] = HeuristicAgent()
            elif agent_type == AgentType.MCTS:
                agents[player] = FastMCTSAgent(
                    iterations=30,   # Ultra-fast iterations
                    time_limit=0.5,  # 500ms timeout
                    exploration_constant=1.414
                )
            elif agent_type == AgentType.HUMAN:
                agents[player] = None  # Human players don't need agents
            else:
                raise HTTPException(status_code=400, detail=f"Unknown agent type: {agent_type}")
        
        self.agent_instances[game_id] = agents
    
    def _start_game(self, game_id: str):
        """Start the game."""
        if game_id not in self.games:
            raise HTTPException(status_code=404, detail="Game not found")
        
        self.games[game_id]['status'] = GameStatus.IN_PROGRESS
        self.games[game_id]['updated_at'] = datetime.now()
        
        # Start the turn loop
        asyncio.create_task(self._run_turn_loop(game_id))
    
    async def _run_turn_loop(self, game_id: str):
        """Run the turn loop for automatic agent moves."""
        logger.info(f"Starting turn loop for game {game_id}")
        while game_id in self.games and self.games[game_id]['status'] == GameStatus.IN_PROGRESS:
            try:
                game_data = self.games[game_id]
                game = game_data['game']
                
                if game.is_game_over():
                    logger.info(f"Game {game_id} is over")
                    await self._end_game(game_id)
                    break
                
                current_player = game.get_current_player()
                agents = self.agent_instances.get(game_id, {})
                agent = agents.get(current_player)
                
                logger.debug(f"Current player: {current_player.name}, Agent: {type(agent).__name__ if agent else 'None'}")
                
                if agent is not None:
                    # Agent move
                    await self._make_agent_move(game_id, current_player, agent)
                else:
                    # Human player - wait for WebSocket move
                    logger.debug(f"Waiting for human move from {current_player.name}")
                    await self._broadcast_game_state(game_id, "waiting_for_human_move")
                    await asyncio.sleep(0.5)  # Longer delay for human players to reduce CPU usage
                
            except Exception as e:
                logger.error(f"Error in turn loop for game {game_id}: {e}")
                import traceback
                traceback.print_exc()
                await self._broadcast_error(game_id, str(e))
                break
    
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
                game.board._update_current_player()
                await self._broadcast_game_state(game_id, "player_skipped")
                return
            
            # Get agent's move with timeout
            start_agent = time.perf_counter()
            try:
                # Run agent selection with a timeout
                move = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, agent.select_action, game.board, player, legal_moves
                    ),
                    timeout=5.0  # 5 second timeout
                )
                end_agent = time.perf_counter()
                agent_time = end_agent - start_agent
                logger.info(f"AGENT MOVE agent_selection: move={move} selected in {agent_time:.4f}s")
            except asyncio.TimeoutError:
                # Agent timed out - pass turn instead of making random move
                logger.warning(f"AGENT MOVE: Agent {type(agent).__name__} timed out after 5 seconds in game {game_id} for player {player.name}")
                game.board._update_current_player()
                await self._broadcast_game_state(game_id, "player_skipped_timeout")
                return
            except Exception as e:
                # Agent raised exception - pass turn instead of making random move
                logger.error(f"AGENT MOVE: Agent {type(agent).__name__} raised exception in game {game_id} for player {player.name}: {e}")
                import traceback
                traceback.print_exc()
                game.board._update_current_player()
                await self._broadcast_game_state(game_id, "player_skipped_error")
                return
            
            # Make the move (only reached if agent successfully returned a move)
            start_apply = time.perf_counter()
            success = game.make_move(move, player)
            end_apply = time.perf_counter()
            apply_time = end_apply - start_apply
            logger.info(f"AGENT MOVE apply: success={success} in {apply_time:.4f}s")
            
            if success:
                # Log the move details
                player_name = self._convert_player_back(player).value
                logger.info(f"Player {player_name} placed a piece at {move.anchor_row},{move.anchor_col} (piece_id={move.piece_id}, orientation={move.orientation})")
                
                # Prepare move information for broadcast
                last_move = {
                    "piece_id": move.piece_id,
                    "orientation": move.orientation,
                    "anchor_row": move.anchor_row,
                    "anchor_col": move.anchor_col,
                    "player": player_name
                }
                
                start_broadcast = time.perf_counter()
                await self._broadcast_game_state(game_id, "move_made", last_move=last_move)
                end_broadcast = time.perf_counter()
                broadcast_time = end_broadcast - start_broadcast
                
                end_total = time.perf_counter()
                total_time = end_total - start_total
                logger.info(f"AGENT MOVE timing: total={total_time:.4f}s, legal={legal_time:.4f}s, agent={agent_time:.4f}s, apply={apply_time:.4f}s, broadcast={broadcast_time:.4f}s")
            else:
                logger.warning(f"AGENT MOVE: Invalid move from agent {type(agent).__name__} for player {player.name}")
                await self._broadcast_error(game_id, "Invalid move from agent")
                
        except Exception as e:
            logger.error(f"AGENT MOVE ERROR: {str(e)} for player {player.name}")
            import traceback
            traceback.print_exc()
            await self._broadcast_error(game_id, f"Agent error: {str(e)}")
    
    async def _end_game(self, game_id: str):
        """End the game."""
        if game_id not in self.games:
            return
        
        self.games[game_id]['status'] = GameStatus.FINISHED
        self.games[game_id]['updated_at'] = datetime.now()
        
        await self._broadcast_game_state(game_id, "game_over")
    
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
        
        # Make the move
        success = game.make_move(engine_move, player)
        
        if success:
            game_data['updated_at'] = datetime.now()
            await self._broadcast_game_state(game_id, "move_made")
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
            
            # Advance to next player (same logic as agent skip)
            game.board._update_current_player()
            logger.info(f"Turn advanced to {game.get_current_player().name}")
            
            # Update game data
            game_data['updated_at'] = datetime.now()
            
            # Broadcast the updated game state
            await self._broadcast_game_state(game_id, "player_passed")
            
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
            
            # Make the move immediately
            start_make_move = time.perf_counter()
            success = game.make_move(engine_move, engine_player)
            end_make_move = time.perf_counter()
            make_move_time = end_make_move - start_make_move
            logger.info(f"HUMAN MOVE make_move: success={success} in {make_move_time:.4f}s")
            
            if success:
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
                
                # Broadcast the updated game state immediately
                start_broadcast = time.perf_counter()
                await self._broadcast_game_state(game_id, "move_made", last_move=last_move)
                end_broadcast = time.perf_counter()
                broadcast_time = end_broadcast - start_broadcast
                
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
            heatmap=heatmap
        )
    
    async def _broadcast_game_state(self, game_id: str, update_type: str, last_move: Optional[Dict[str, Any]] = None):
        """Broadcast game state to all WebSocket connections."""
        if game_id not in self.websocket_connections:
            return
        
        game_state = self._get_game_state(game_id)
        data = {"game_state": game_state.dict()}
        
        # Include last move information if provided
        if last_move:
            data["move"] = last_move
            data["player"] = last_move.get("player")
        
        update = StateUpdate(
            type=update_type,
            game_id=game_id,
            data=data
        )
        
        message = update.json()
        
        # Send to all connections
        disconnected = []
        for websocket in self.websocket_connections[game_id]:
            try:
                await websocket.send_text(message)
            except:
                disconnected.append(websocket)
        
        # Remove disconnected connections
        for ws in disconnected:
            self.websocket_connections[game_id].remove(ws)
    
    async def _broadcast_error(self, game_id: str, error_message: str):
        """Broadcast error to all WebSocket connections."""
        if game_id not in self.websocket_connections:
            return
        
        update = StateUpdate(
            type="error",
            game_id=game_id,
            data={"error": error_message}
        )
        
        message = update.json()
        
        # Send to all connections
        disconnected = []
        for websocket in self.websocket_connections[game_id]:
            try:
                await websocket.send_text(message)
            except:
                disconnected.append(websocket)
        
        # Remove disconnected connections
        for ws in disconnected:
            self.websocket_connections[game_id].remove(ws)
    
    def add_websocket_connection(self, game_id: str, websocket: WebSocket):
        """Add a WebSocket connection for a game."""
        if game_id not in self.websocket_connections:
            self.websocket_connections[game_id] = []
        self.websocket_connections[game_id].append(websocket)
    
    def remove_websocket_connection(self, game_id: str, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if game_id in self.websocket_connections:
            try:
                self.websocket_connections[game_id].remove(websocket)
            except ValueError:
                pass
    
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


# Global game manager instance
game_manager = GameManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown tasks:
    - Startup: Connect to MongoDB
    - Shutdown: Close MongoDB connection gracefully
    """
    # Startup
    logger.info("Starting Blokus RL Web API...")
    try:
        await connect_to_mongo()
        logger.info("MongoDB connection established")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB during startup: {e}")
        logger.warning("Server will start without MongoDB connection. Some features may be unavailable.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Blokus RL Web API...")
    await close_mongo_connection()


# Create FastAPI app
app = FastAPI(
    title="Blokus RL Web API",
    description="Web API for Blokus RL game with REST and WebSocket support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# REST API Endpoints

@app.post("/api/games", response_model=GameCreateResponse)
async def create_game(config: GameConfig):
    """Create a new game."""
    try:
        game_id = game_manager.create_game(config)
        game_state = game_manager.get_game_state(game_id)
        
        return GameCreateResponse(
            game_id=game_id,
            game_state=game_state,
            message="Game created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information."""
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


@app.get("/api/health/db")
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


@app.get("/api/games/{game_id}", response_model=GameState)
async def get_game(game_id: str):
    """Get game state."""
    return game_manager.get_game_state(game_id)


@app.post("/api/games/{game_id}/move", response_model=MoveResponse)
async def make_move(game_id: str, move_request: MoveRequest):
    """Make a move in the game."""
    return await game_manager.make_move(game_id, move_request)


@app.get("/api/agents", response_model=List[AgentInfo])
async def get_agents():
    """Get list of available agents."""
    return game_manager.get_available_agents()


@app.get("/api/games", response_model=List[GameState])
async def list_games():
    """List all active games."""
    games = []
    for game_id in game_manager.games:
        games.append(game_manager.get_game_state(game_id))
    return games


# Training Runs API Endpoints

@app.get("/api/training-runs")
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


@app.get("/api/training-runs/{run_id}")
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


@app.get("/api/training-runs/agents/list")
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


@app.get("/api/training-runs/{run_id}/evaluations")
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


# WebSocket Endpoint

@app.websocket("/ws/games/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """WebSocket endpoint for real-time game updates."""
    await websocket.accept()
    
    # Check if game exists
    if game_id not in game_manager.games:
        await websocket.send_text(json.dumps({
            "type": "error",
            "game_id": game_id,
            "data": {"error": "Game not found"},
            "timestamp": datetime.now().isoformat()
        }))
        await websocket.close()
        return
    
    # Add connection
    game_manager.add_websocket_connection(game_id, websocket)
    logger.info(f"Connected to game {game_id}")
    
    # Send initial game state
    game_state = game_manager.get_game_state(game_id)
    initial_update = StateUpdate(
        type="game_state",
        game_id=game_id,
        data={"game_state": game_state.dict()}
    )
    await websocket.send_text(initial_update.json())
    logger.info("Game state updated")
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "move":
                logger.info(f"Received move message for game {game_id}: {json.dumps(message)}")
                move_data = message.get("data")
                if move_data:
                    try:
                        logger.info(f"Creating MoveRequest from data: {json.dumps(move_data)}")
                        move_request = MoveRequest(**move_data)
                        logger.info(f"MoveRequest created: player={move_request.player}, move={move_request.move.dict() if hasattr(move_request.move, 'dict') else move_request.move}")
                        
                        # Process move immediately without waiting for turn loop
                        response = await game_manager._process_human_move_immediately(game_id, move_request)
                        logger.info(f"Sending move response: success={response.success}")
                        await websocket.send_text(response.json())
                    except Exception as e:
                        logger.error(f"Error processing move: {e}")
                        import traceback
                        traceback.print_exc()
                        error_response = MoveResponse(
                            success=False,
                            message=f"Invalid move data: {str(e)}",
                            game_over=False
                        )
                        await websocket.send_text(error_response.json())
                else:
                    logger.warning("No move data in message")
                    error_response = MoveResponse(
                        success=False,
                        message="No move data provided",
                        game_over=False
                    )
                    await websocket.send_text(error_response.json())
            
            elif message.get("type") == "pass":
                logger.info(f"Received pass message for game {game_id}: {json.dumps(message)}")
                pass_data = message.get("data")
                if pass_data and "player" in pass_data:
                    try:
                        # Extract player from data
                        player_str = pass_data.get("player")
                        # Convert to Player enum
                        player = Player(player_str)
                        
                        # Process pass immediately
                        response = await game_manager._process_human_pass(game_id, player)
                        logger.info(f"Sending pass response: success={response.success}")
                        await websocket.send_text(response.json())
                    except Exception as e:
                        logger.error(f"Error processing pass: {e}")
                        import traceback
                        traceback.print_exc()
                        error_response = MoveResponse(
                            success=False,
                            message=f"Invalid pass data: {str(e)}",
                            game_over=False
                        )
                        await websocket.send_text(error_response.json())
                else:
                    logger.warning("No player data in pass message")
                    error_response = MoveResponse(
                        success=False,
                        message="No player data provided",
                        game_over=False
                    )
                    await websocket.send_text(error_response.json())
            
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            
    except WebSocketDisconnect:
        game_manager.remove_websocket_connection(game_id, websocket)
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "game_id": game_id,
            "data": {"error": str(e)},
            "timestamp": datetime.now().isoformat()
        }))
        game_manager.remove_websocket_connection(game_id, websocket)


# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP Error",
            message=exc.detail
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred"
        ).dict()
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)