"""
Game manager for handling multiple concurrent Blokus games.
"""

import asyncio
import uuid
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from engine.board import Board, Player as EnginePlayer
from engine.game import BlokusGame
from engine.move_generator import LegalMoveGenerator
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from mcts.mcts_agent import MCTSAgent
from schemas.game_config import GameConfig, AgentConfig, PlayerType
from schemas.move import MoveRequest, MoveResponse, Player as SchemaPlayer
from schemas.state_update import GameState, PlayerState, BoardState, LegalMove, Position


@dataclass
class GameSession:
    """Represents an active game session."""
    game_id: str
    game: BlokusGame
    config: GameConfig
    agents: Dict[SchemaPlayer, Any]  # Player -> Agent instance
    move_generator: LegalMoveGenerator
    created_at: float
    last_updated: float
    status: str = "active"  # "active", "completed", "error"
    error_message: Optional[str] = None


class GameManager:
    """Manages multiple concurrent Blokus games."""
    
    def __init__(self):
        """Initialize game manager."""
        self.games: Dict[str, GameSession] = {}
        self.move_generator = LegalMoveGenerator()
        
    def create_game(self, config: GameConfig) -> str:
        """
        Create a new game.
        
        Args:
            config: Game configuration
            
        Returns:
            Game ID
        """
        game_id = str(uuid.uuid4())
        
        # Create game instance
        game = BlokusGame()
        
        # Create agents
        agents = self._create_agents(config.players)
        
        # Create game session
        session = GameSession(
            game_id=game_id,
            game=game,
            config=config,
            agents=agents,
            move_generator=self.move_generator,
            created_at=time.time(),
            last_updated=time.time()
        )
        
        self.games[game_id] = session
        
        return game_id
        
    def _create_agents(self, player_configs: List[AgentConfig]) -> Dict[SchemaPlayer, Any]:
        """Create agent instances from configuration."""
        agents = {}
        engine_players = [EnginePlayer.RED, EnginePlayer.BLUE, EnginePlayer.GREEN, EnginePlayer.YELLOW]
        
        for i, player_config in enumerate(player_configs):
            if i >= len(engine_players):
                break
                
            engine_player = engine_players[i]
            schema_player = self._engine_to_schema_player(engine_player)
            
            if player_config.type == PlayerType.RANDOM:
                agent = RandomAgent(seed=player_config.seed)
            elif player_config.type == PlayerType.HEURISTIC:
                agent = HeuristicAgent(seed=player_config.seed)
                if player_config.parameters:
                    agent.set_weights(player_config.parameters)
            elif player_config.type == PlayerType.MCTS:
                agent = MCTSAgent(
                    iterations=player_config.parameters.get("iterations", 1000),
                    time_limit=player_config.parameters.get("time_limit"),
                    exploration_constant=player_config.parameters.get("exploration_constant", 1.414),
                    use_transposition_table=player_config.parameters.get("use_transposition_table", True),
                    seed=player_config.seed
                )
            else:  # HUMAN
                agent = None  # Human players don't have agents
                
            agents[schema_player] = agent
            
        return agents
        
    def get_game(self, game_id: str) -> Optional[GameSession]:
        """Get game session by ID."""
        return self.games.get(game_id)
        
    def get_game_state(self, game_id: str) -> Optional[GameState]:
        """
        Get current game state.
        
        Args:
            game_id: Game ID
            
        Returns:
            Game state or None if game not found
        """
        session = self.get_game(game_id)
        if not session:
            return None
            
        return self._create_game_state(session)
        
    def make_move(self, game_id: str, move_request: MoveRequest) -> MoveResponse:
        """
        Make a move in a game.
        
        Args:
            game_id: Game ID
            move_request: Move request
            
        Returns:
            Move response
        """
        session = self.get_game(game_id)
        if not session:
            return MoveResponse(
                success=False,
                message="Game not found",
                game_over=False
            )
            
        if session.status != "active":
            return MoveResponse(
                success=False,
                message=f"Game is {session.status}",
                game_over=True
            )
            
        # Convert schema player to engine player
        engine_player = self._schema_to_engine_player(move_request.player)
        
        # Check if it's the player's turn
        if session.game.get_current_player() != engine_player:
            return MoveResponse(
                success=False,
                message="Not your turn",
                game_over=False
            )
            
        # Create move object
        from engine.move_generator import Move
        move = Move(
            piece_id=move_request.piece_id,
            orientation=move_request.orientation,
            anchor_row=move_request.anchor_row,
            anchor_col=move_request.anchor_col
        )
        
        # Make the move
        try:
            success = session.game.make_move(move, engine_player)
            
            if success:
                session.last_updated = time.time()
                
                # Check if game is over
                game_over = session.game.is_game_over()
                winner = None
                if game_over:
                    winner_engine = session.game.board.get_winner()
                    if winner_engine:
                        winner = self._engine_to_schema_player(winner_engine)
                    session.status = "completed"
                    
                return MoveResponse(
                    success=True,
                    message="Move successful",
                    new_score=session.game.get_score(engine_player),
                    game_over=game_over,
                    winner=winner
                )
            else:
                return MoveResponse(
                    success=False,
                    message="Invalid move",
                    game_over=False
                )
                
        except Exception as e:
            session.status = "error"
            session.error_message = str(e)
            return MoveResponse(
                success=False,
                message=f"Error making move: {str(e)}",
                game_over=True
            )
            
    def get_legal_moves(self, game_id: str, player: SchemaPlayer) -> List[LegalMove]:
        """Get legal moves for a player."""
        session = self.get_game(game_id)
        if not session:
            return []
            
        engine_player = self._schema_to_engine_player(player)
        moves = session.move_generator.get_legal_moves(session.game.board, engine_player)
        
        legal_moves = []
        for move in moves:
            # Get positions this move would occupy
            orientations = session.move_generator.piece_orientations_cache[move.piece_id]
            orientation = orientations[move.orientation]
            
            positions = []
            for i in range(orientation.shape[0]):
                for j in range(orientation.shape[1]):
                    if orientation[i, j] == 1:
                        positions.append(Position(
                            row=move.anchor_row + i,
                            col=move.anchor_col + j
                        ))
                        
            legal_moves.append(LegalMove(
                piece_id=move.piece_id,
                orientation=move.orientation,
                anchor_row=move.anchor_row,
                anchor_col=move.anchor_col,
                positions=positions
            ))
            
        return legal_moves
        
    def _create_game_state(self, session: GameSession) -> GameState:
        """Create game state from session."""
        # Create board state
        board_cells = []
        for row in range(20):
            board_row = []
            for col in range(20):
                cell_value = session.game.board.get_cell(session.game.board.Position(row, col))
                if cell_value == 0:
                    board_row.append(None)
                else:
                    board_row.append(self._engine_to_schema_player(EnginePlayer(cell_value)))
            board_cells.append(board_row)
            
        board_state = BoardState(
            cells=board_cells,
            move_count=session.game.move_count
        )
        
        # Create player states
        players = []
        engine_players = [EnginePlayer.RED, EnginePlayer.BLUE, EnginePlayer.GREEN, EnginePlayer.YELLOW]
        
        for engine_player in engine_players:
            schema_player = self._engine_to_schema_player(engine_player)
            score = session.game.get_score(engine_player)
            pieces_used = list(session.game.board.player_pieces_used[engine_player])
            pieces_remaining = [i for i in range(1, 22) if i not in pieces_used]
            
            players.append(PlayerState(
                player=schema_player,
                score=score,
                pieces_used=pieces_used,
                pieces_remaining=pieces_remaining,
                is_active=(session.game.get_current_player() == engine_player)
            ))
            
        # Get legal moves for current player
        current_engine_player = session.game.get_current_player()
        current_schema_player = self._engine_to_schema_player(current_engine_player)
        legal_moves = self.get_legal_moves(session.game_id, current_schema_player)
        
        # Calculate heatmap: map legal_moves to 20x20 grid (1 = legal, 0 = illegal)
        heatmap = [[0.0 for _ in range(20)] for _ in range(20)]
        for legal_move in legal_moves:
            # Mark all positions of this legal move as 1.0
            for position in legal_move.positions:
                if 0 <= position.row < 20 and 0 <= position.col < 20:
                    heatmap[position.row][position.col] = 1.0
        
        # Check game over
        game_over = session.game.is_game_over()
        winner = None
        if game_over:
            winner_engine = session.game.board.get_winner()
            if winner_engine:
                winner = self._engine_to_schema_player(winner_engine)
                
        return GameState(
            game_id=session.game_id,
            current_player=current_schema_player,
            board=board_state,
            players=players,
            legal_moves=legal_moves,
            game_over=game_over,
            winner=winner,
            last_move=None,  # TODO: Track last move
            heatmap=heatmap
        )
        
    def _engine_to_schema_player(self, engine_player: EnginePlayer) -> SchemaPlayer:
        """Convert engine player to schema player."""
        mapping = {
            EnginePlayer.RED: SchemaPlayer.RED,
            EnginePlayer.BLUE: SchemaPlayer.BLUE,
            EnginePlayer.GREEN: SchemaPlayer.GREEN,
            EnginePlayer.YELLOW: SchemaPlayer.YELLOW
        }
        return mapping[engine_player]
        
    def _schema_to_engine_player(self, schema_player: SchemaPlayer) -> EnginePlayer:
        """Convert schema player to engine player."""
        mapping = {
            SchemaPlayer.RED: EnginePlayer.RED,
            SchemaPlayer.BLUE: EnginePlayer.BLUE,
            SchemaPlayer.GREEN: EnginePlayer.GREEN,
            SchemaPlayer.YELLOW: EnginePlayer.YELLOW
        }
        return mapping[schema_player]
        
    def cleanup_old_games(self, max_age_hours: int = 24):
        """Clean up old games."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        games_to_remove = []
        for game_id, session in self.games.items():
            if current_time - session.last_updated > max_age_seconds:
                games_to_remove.append(game_id)
                
        for game_id in games_to_remove:
            del self.games[game_id]
            
        return len(games_to_remove)
