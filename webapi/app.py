"""
FastAPI application for Blokus RL Web API
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

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
        print(f"🔄 Starting turn loop for game {game_id}")
        while game_id in self.games and self.games[game_id]['status'] == GameStatus.IN_PROGRESS:
            try:
                game_data = self.games[game_id]
                game = game_data['game']
                
                if game.is_game_over():
                    print(f"🏁 Game {game_id} is over")
                    await self._end_game(game_id)
                    break
                
                current_player = game.get_current_player()
                agents = self.agent_instances.get(game_id, {})
                agent = agents.get(current_player)
                
                print(f"🎮 Current player: {current_player}, Agent: {type(agent).__name__ if agent else 'None'}")
                
                if agent is not None:
                    # Agent move
                    print(f"🤖 Making agent move for {current_player}")
                    await self._make_agent_move(game_id, current_player, agent)
                else:
                    # Human player - wait for WebSocket move
                    print(f"👤 Waiting for human move from {current_player}")
                    await self._broadcast_game_state(game_id, "waiting_for_human_move")
                    await asyncio.sleep(0.5)  # Longer delay for human players to reduce CPU usage
                
            except Exception as e:
                print(f"❌ Error in turn loop for game {game_id}: {e}")
                import traceback
                traceback.print_exc()
                await self._broadcast_error(game_id, str(e))
                break
    
    async def _make_agent_move(self, game_id: str, player: EnginePlayer, agent: Any):
        """Make a move for an agent."""
        try:
            print(f"🤖 Making agent move for {player} in game {game_id}")
            game_data = self.games[game_id]
            game = game_data['game']
            
            # Get legal moves
            legal_moves = game.get_legal_moves(player)
            print(f"📋 Found {len(legal_moves)} legal moves for {player}")
            
            if not legal_moves:
                # Player cannot move, skip turn
                print(f"⏭️ No legal moves for {player}, skipping turn")
                game.board._update_current_player()
                await self._broadcast_game_state(game_id, "player_skipped")
                return
            
            # Get agent's move with timeout
            print(f"🎯 Agent {type(agent).__name__} selecting move...")
            try:
                # Run agent selection with a timeout
                move = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, agent.select_action, game.board, player, legal_moves
                    ),
                    timeout=2.0  # 2 second timeout (reduced from 10)
                )
                print(f"🎲 Agent selected move: {move}")
            except asyncio.TimeoutError:
                print(f"⏰ Agent {type(agent).__name__} timed out, selecting random move")
                # Fallback to random move if agent times out
                import random
                move = random.choice(legal_moves)
                print(f"🎲 Random fallback move: {move}")
            except Exception as e:
                print(f"❌ Agent error, using random fallback: {e}")
                import random
                move = random.choice(legal_moves)
                print(f"🎲 Error fallback move: {move}")
            
            # Make the move
            success = game.make_move(move, player)
            print(f"✅ Move successful: {success}")
            
            if success:
                await self._broadcast_game_state(game_id, "move_made")
            else:
                await self._broadcast_error(game_id, "Invalid move from agent")
                
        except Exception as e:
            print(f"❌ Agent error for {player}: {str(e)}")
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
    
    async def _process_human_move_immediately(self, game_id: str, move_request: MoveRequest) -> MoveResponse:
        """Process human move immediately without waiting for turn loop."""
        try:
            print(f"⚡ Processing human move immediately for {move_request.player}")
            
            # Get game data
            if game_id not in self.games:
                return MoveResponse(success=False, message="Game not found", game_over=False)
            
            game_data = self.games[game_id]
            game = game_data['game']
            
            # Convert schema player to engine player
            engine_player = self._convert_player(move_request.player)
            
            # Verify it's the player's turn
            current_player = game.get_current_player()
            if current_player != engine_player:
                return MoveResponse(
                    success=False, 
                    message=f"It's not {move_request.player}'s turn", 
                    game_over=False
                )
            
            # Create move object
            move_data = move_request.move
            engine_move = EngineMove(move_data.piece_id, move_data.orientation, move_data.anchor_row, move_data.anchor_col)
            
            # Make the move immediately
            success = game.make_move(engine_move, engine_player)
            print(f"⚡ Human move result: {success}")
            
            if success:
                # Update game data
                game_data['updated_at'] = datetime.now()
                
                # Broadcast the updated game state immediately
                await self._broadcast_game_state(game_id, "move_made")
                
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
                return MoveResponse(
                    success=False,
                    message="Invalid move",
                    game_state=self._get_game_state(game_id)
                )
                
        except Exception as e:
            print(f"❌ Error in immediate move processing: {e}")
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
        
        # Get legal moves
        legal_moves = []
        if not game.is_game_over():
            engine_moves = game.get_legal_moves()
            legal_moves = [self._convert_move_back(move) for move in engine_moves]
        
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
            updated_at=game_data['updated_at']
        )
    
    async def _broadcast_game_state(self, game_id: str, update_type: str):
        """Broadcast game state to all WebSocket connections."""
        if game_id not in self.websocket_connections:
            return
        
        game_state = self._get_game_state(game_id)
        update = StateUpdate(
            type=update_type,
            game_id=game_id,
            data={"game_state": game_state.dict()}
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
    """Application lifespan manager."""
    # Startup
    print("Starting Blokus RL Web API...")
    yield
    # Shutdown
    print("Shutting down Blokus RL Web API...")


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
    allow_origins=["*"],
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
            "websocket": "/ws/games/{game_id}"
        }
    }

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
    
    # Send initial game state
    game_state = game_manager.get_game_state(game_id)
    initial_update = StateUpdate(
        type="game_state",
        game_id=game_id,
        data={"game_state": game_state.dict()}
    )
    await websocket.send_text(initial_update.json())
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "move":
                print(f"🎮 Received move message for game {game_id}: {message}")
                move_data = message.get("data")
                if move_data:
                    try:
                        print(f"📝 Creating MoveRequest from data: {move_data}")
                        move_request = MoveRequest(**move_data)
                        print(f"✅ MoveRequest created: {move_request}")
                        
                        # Process move immediately without waiting for turn loop
                        response = await game_manager._process_human_move_immediately(game_id, move_request)
                        print(f"📤 Sending response: {response}")
                        await websocket.send_text(response.json())
                    except Exception as e:
                        print(f"❌ Error processing move: {e}")
                        error_response = MoveResponse(
                            success=False,
                            message=f"Invalid move data: {str(e)}",
                            game_over=False
                        )
                        print(f"📤 Sending error response: {error_response}")
                        await websocket.send_text(error_response.json())
                else:
                    print("❌ No move data in message")
                    error_response = MoveResponse(
                        success=False,
                        message="No move data provided",
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