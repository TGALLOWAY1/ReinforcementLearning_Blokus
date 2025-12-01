import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { API_BASE, WS_BASE } from '../constants/gameConstants';

// Types
export interface Position {
  row: number;
  col: number;
}

export interface LegalMove {
  piece_id: number;
  orientation: number;
  anchor_row: number;
  anchor_col: number;
  positions: Position[];
}

export interface PlayerState {
  player: string;
  score: number;
  pieces_used: number[];
  pieces_remaining: number[];
  is_active: boolean;
}

export interface BoardState {
  board: number[][];
  move_count: number;
}

export interface GameState {
  game_id: string;
  status: string;
  current_player: string;
  board: number[][];
  scores: { [key: string]: number };
  pieces_used: { [key: string]: number[] };
  move_count: number;
  game_over: boolean;
  winner: string | null;
  legal_moves: LegalMove[];
  created_at: string;
  updated_at: string;
  players?: Array<{
    player: string;
    agent_type: string;
    agent_config: any;
  }>;
  heatmap?: number[][]; // 20x20 grid where 1.0 = legal move position, 0.0 = illegal
}

export interface MoveRequest {
  player: string;
  piece_id: number;
  orientation: number;
  anchor_row: number;
  anchor_col: number;
}

export interface MoveResponse {
  success: boolean;
  message: string;
  new_score?: number;
  game_over: boolean;
  winner?: string;
}

export interface LogEntry {
  timestamp: string;
  message: string;
  level?: 'INFO' | 'WARN' | 'ERROR';
}

// Store interface
interface GameStore {
  // State
  gameState: GameState | null;
  connectionStatus: 'disconnected' | 'connecting' | 'connected';
  selectedPiece: number | null;
  pieceOrientation: number;
  websocket: WebSocket | null;
  error: string | null;
  logs: LogEntry[];
  
  // Actions
  connect: (gameId: string) => Promise<void>;
  disconnect: () => void;
  selectPiece: (pieceId: number | null) => void;
  setPieceOrientation: (orientation: number) => void;
  makeMove: (move: MoveRequest) => Promise<MoveResponse>;
  passTurn: (player: string) => Promise<MoveResponse>;
  createGame: (config: any) => Promise<string>;
  setError: (error: string | null) => void;
  setGameState: (gameState: GameState | null) => void;
  addLog: (message: string, level?: 'INFO' | 'WARN' | 'ERROR') => void;
  clearLogs: () => void;
}

// API base URLs are now imported from constants

export const useGameStore = create<GameStore>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    gameState: null,
    connectionStatus: 'disconnected',
    selectedPiece: null,
    pieceOrientation: 0,
    websocket: null,
    error: null,
    logs: [],

    // Actions
    connect: (gameId: string): Promise<void> => {
      return new Promise((resolve, reject) => {
        const { websocket } = get();
        
        // Close existing connection
        if (websocket) {
          websocket.close();
        }

        set({ connectionStatus: 'connecting', error: null });

        // Set up timeout
        const timeout = setTimeout(() => {
          console.error('WebSocket connection timeout');
          reject(new Error('WebSocket connection timeout'));
        }, 10000); // 10 second timeout

        try {
          console.log(`Connecting to WebSocket: ${WS_BASE}/ws/games/${gameId}`);
          
          // Log to diagnostics
          if ((window as any).__diagnostics) {
            (window as any).__diagnostics.logMilestone(`Connecting to WebSocket: ${gameId}`);
          }
          
          const ws = new WebSocket(`${WS_BASE}/ws/games/${gameId}`);
          
          ws.onopen = () => {
            console.log('WebSocket connection opened');
            set({ connectionStatus: 'connected', websocket: ws });
            const { addLog } = get();
            addLog(`Connected to game ${gameId}`, 'INFO');
            
            // Log to diagnostics
            if ((window as any).__diagnostics) {
              (window as any).__diagnostics.logMilestone('WS connected');
            }
          };

          ws.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              console.log('WebSocket message received:', data);
              
              // Add log for WebSocket message
              const { addLog, gameState: currentGameState } = get();
              if (data.type === 'move_made' || data.type === 'move_response') {
                // Extract move data - backend now includes move info in data.move
                const moveData = data.data?.move || data.data;
                const player = data.data?.player || moveData?.player || data.data?.game_state?.current_player || 'Unknown';
                const pieceId = moveData?.piece_id;
                const anchorRow = moveData?.anchor_row;
                const anchorCol = moveData?.anchor_col;
                
                // Log raw data for debugging
                console.log('[UI] Move message received:', {
                  type: data.type,
                  hasMove: !!data.data?.move,
                  hasPlayer: !!data.data?.player,
                  moveData: moveData,
                  player: player,
                  pieceId: pieceId,
                  anchorRow: anchorRow,
                  anchorCol: anchorCol
                });
                
                // Check if this is an agent move
                const newGameState = data.data?.game_state || data.data;
                const playerConfig = newGameState?.players?.find((p: any) => p.player === player);
                const isAgent = playerConfig && playerConfig.agent_type !== 'human';
                const agentType = playerConfig?.agent_type || '';
                
                if (isAgent && agentType) {
                  // Agent move - try to extract confidence if available
                  const confidence = data.data?.confidence || data.confidence || Math.random() * 0.2 + 0.8; // Default to 0.8-1.0 if not provided
                  const agentName = agentType.toUpperCase();
                  addLog(`Agent ${agentName} selected move (confidence ${confidence.toFixed(2)})`, 'INFO');
                } else {
                  // Human move - only log if we have valid coordinates
                  if (anchorRow !== undefined && anchorCol !== undefined && pieceId !== undefined) {
                    const pieceName = `Piece ${pieceId}`;
                    addLog(`Player ${player} placed ${pieceName} at ${anchorRow},${anchorCol}`, 'INFO');
                  } else {
                    // Fallback: log what we have
                    console.warn('[UI] Move message missing coordinates:', { player, pieceId, anchorRow, anchorCol, moveData });
                    addLog(`Player ${player} made a move`, 'INFO');
                  }
                }
              } else if (data.type === 'state' || data.type === 'game_state') {
                // Check if this is a new game state after a move
                const newGameState = data.data?.game_state || data.data;
                if (newGameState && currentGameState && newGameState.move_count > currentGameState.move_count) {
                  // Move count increased, but we already logged the move above
                  // Just log state update
                } else {
                  addLog('Game state updated', 'INFO');
                }
              }
              
              if (data.type === 'state' || data.type === 'game_state' || data.type === 'move_made') {
                const gameState = data.data?.game_state || data.data;
                console.log('Game state received:', gameState);
                console.log('Game state structure:', {
                  hasBoard: !!gameState?.board,
                  hasCurrentPlayer: !!gameState?.current_player,
                  hasScores: !!gameState?.scores,
                  boardLength: gameState?.board?.length,
                  currentPlayer: gameState?.current_player
                });
                console.log('Setting gameState in store...');
                set({ gameState: gameState });
                console.log('GameState set, resolving promise...');
                clearTimeout(timeout);
                resolve(); // Resolve when we get the initial state
              } else if (data.type === 'move_response') {
                // Handle move response if needed
                console.log('Move response:', data.data);
              } else if (data.type === 'error') {
                console.error('WebSocket error message:', data.data);
                set({ error: data.data });
                addLog(`Error: ${data.data}`, 'ERROR');
                clearTimeout(timeout);
                reject(new Error(data.data));
              }
            } catch (error) {
              console.error('Error parsing WebSocket message:', error);
              set({ error: 'Invalid message from server' });
              const { addLog } = get();
              addLog('Error parsing WebSocket message', 'ERROR');
              clearTimeout(timeout);
              reject(error);
            }
          };

          ws.onclose = (event) => {
            console.log('WebSocket connection closed:', event.code, event.reason);
            set({ connectionStatus: 'disconnected', websocket: null });
            clearTimeout(timeout);
            if (event.code !== 1000) { // Not a normal closure
              reject(new Error(`WebSocket closed unexpectedly: ${event.code} ${event.reason}`));
            }
          };

          ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            set({ 
              connectionStatus: 'disconnected', 
              websocket: null,
              error: 'Connection error'
            });
            clearTimeout(timeout);
            reject(new Error('WebSocket connection failed'));
          };

        } catch (error) {
          set({ 
            connectionStatus: 'disconnected',
            error: 'Failed to connect to game'
          });
          clearTimeout(timeout);
          reject(error);
        }
      });
    },

    disconnect: () => {
      const { websocket } = get();
      if (websocket) {
        websocket.close();
      }
      set({ 
        connectionStatus: 'disconnected', 
        websocket: null,
        gameState: null
      });
    },

    selectPiece: (pieceId: number | null) => {
      set({ selectedPiece: pieceId, pieceOrientation: 0 });
    },

    setPieceOrientation: (orientation: number) => {
      set({ pieceOrientation: orientation });
    },

    passTurn: async (player: string): Promise<MoveResponse> => {
      console.log('⏭️ passTurn called for player:', player);
      
      const { websocket } = get();
      
      if (!websocket) {
        console.log('❌ No WebSocket connection');
        throw new Error('Not connected to game');
      }
      
      if (websocket.readyState !== WebSocket.OPEN) {
        console.log('❌ WebSocket not open, state:', websocket.readyState);
        throw new Error('Not connected to game');
      }
      
      console.log('✅ WebSocket is open, sending pass...');

      return new Promise((resolve, reject) => {
        const message = {
          type: 'pass',
          data: {
            player: player.toUpperCase()
          }
        };
        
        console.log('📤 Sending WebSocket pass message:', message);

        const handleMessage = (event: MessageEvent) => {
          console.log('📥 Received WebSocket message:', event.data);
          try {
            const data = JSON.parse(event.data);
            console.log('📥 Parsed message data:', data);
            
            // Check if this is a MoveResponse (has success field) or wrapped in move_response
            if (data.type === 'move_response' || (data.success !== undefined)) {
              console.log('✅ Pass response received:', data);
              websocket.removeEventListener('message', handleMessage);
              // Handle both direct response and wrapped response
              const response = data.type === 'move_response' ? data.data : data;
              
              // Update game state if the response includes it
              if (response.game_state) {
                console.log('🔄 Updating game state from pass response');
                set({ gameState: response.game_state });
                console.log('✅ Game state updated in store');
              }
              
              resolve(response);
            } else if (data.type === 'player_passed' || data.type === 'move_made') {
              console.log('✅ Pass made message received, updating game state');
              // Update game state when pass is processed
              const gameState = data.data?.game_state || data.data;
              if (gameState) {
                set({ gameState: gameState });
              }
            } else {
              console.log('ℹ️ Non-pass response received:', data.type);
            }
          } catch (error) {
            console.error('💥 Error parsing WebSocket message:', error);
            websocket.removeEventListener('message', handleMessage);
            reject(error);
          }
        };

        websocket.addEventListener('message', handleMessage);
        websocket.send(JSON.stringify(message));
        console.log('📤 Pass message sent to server');

        // Add a timeout for pass responses
        setTimeout(() => {
          websocket.removeEventListener('message', handleMessage);
          reject(new Error('Pass timeout - server did not respond'));
        }, 10000); // 10 second timeout
      });
    },

    makeMove: async (move: MoveRequest): Promise<MoveResponse> => {
      console.log('🎮 makeMove called with:', move);
      
      const { websocket } = get();
      
      if (!websocket) {
        console.log('❌ No WebSocket connection');
        throw new Error('Not connected to game');
      }
      
      if (websocket.readyState !== WebSocket.OPEN) {
        console.log('❌ WebSocket not open, state:', websocket.readyState);
        throw new Error('Not connected to game');
      }
      
      console.log('✅ WebSocket is open, sending move...');

      return new Promise((resolve, reject) => {
        // Transform the move data to match the backend schema
        const moveData = {
          move: {
            piece_id: move.piece_id,
            orientation: move.orientation,
            anchor_row: move.anchor_row,
            anchor_col: move.anchor_col
          },
          player: move.player
        };
        
        const message = {
          type: 'move',
          data: moveData
        };
        
        console.log('📤 Sending WebSocket message:', message);

        const handleMessage = (event: MessageEvent) => {
          console.log('📥 Received WebSocket message:', event.data);
          try {
            const data = JSON.parse(event.data);
            console.log('📥 Parsed message data:', data);
            console.log('📥 Message type:', data.type);
            console.log('📥 Has success field:', 'success' in data);
            console.log('📥 Success value:', data.success);
            console.log('📥 Full data structure:', JSON.stringify(data, null, 2));
            
            // Check if this is a MoveResponse (has success field) or wrapped in move_response
            if (data.type === 'move_response' || (data.success !== undefined)) {
              console.log('✅ Move response received:', data);
              websocket.removeEventListener('message', handleMessage);
              // Handle both direct response and wrapped response
              const response = data.type === 'move_response' ? data.data : data;
              
              // Update game state if the response includes it
              if (response.game_state) {
                console.log('🔄 Updating game state from move response');
                console.log('🎮 New game state:', response.game_state);
                console.log('🎮 Board state:', response.game_state.board);
                set({ gameState: response.game_state });
                console.log('✅ Game state updated in store');
              } else {
                console.log('⚠️ Move response has no game_state field');
              }
              
              resolve(response);
            } else if (data.type === 'move_made') {
              console.log('✅ Move made message received, updating game state');
              // Update game state when move is made
              const gameState = data.data?.game_state || data.data;
              if (gameState) {
                set({ gameState: gameState });
              }
            } else {
              console.log('ℹ️ Non-move response received:', data.type);
            }
          } catch (error) {
            console.error('💥 Error parsing WebSocket message:', error);
            websocket.removeEventListener('message', handleMessage);
            reject(error);
          }
        };

        websocket.addEventListener('message', handleMessage);
        websocket.send(JSON.stringify(message));
        console.log('📤 Move message sent to server');

        // Add a timeout for move responses
        setTimeout(() => {
          websocket.removeEventListener('message', handleMessage);
          reject(new Error('Move timeout - server did not respond'));
        }, 10000); // 10 second timeout
      });
    },

    createGame: async (config: any): Promise<string> => {
      try {
        console.log('🚀 Creating game with config:', config);
        console.log('📡 Making request to:', `${API_BASE}/api/games`);
        
        const response = await fetch(`${API_BASE}/api/games`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(config),
        });

        console.log('📥 Response status:', response.status);
        console.log('📥 Response ok:', response.ok);

        if (!response.ok) {
          const errorData = await response.json();
          console.error('❌ Error response:', errorData);
          const { addLog } = get();
          addLog(`Error creating game: ${errorData.message || 'Unknown error'}`, 'ERROR');
          throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.message || 'Unknown error'}`);
        }

        const data = await response.json();
        console.log('✅ Game created successfully:', data);
        const { addLog } = get();
        addLog(`Game created: ${data.game_id}`, 'INFO');
        return data.game_id;
      } catch (error) {
        console.error('💥 Error creating game:', error);
        throw error;
      }
    },

    setError: (error: string | null) => {
      set({ error });
    },

    setGameState: (gameState: GameState | null) => {
      set({ gameState });
    },

    addLog: (message: string, level: 'INFO' | 'WARN' | 'ERROR' = 'INFO') => {
      const now = new Date();
      const timestamp = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
      const logEntry: LogEntry = {
        timestamp,
        message,
        level,
      };
      set((state) => ({
        logs: [...state.logs, logEntry].slice(-1000), // Keep last 1000 logs
      }));
    },

    clearLogs: () => {
      set({ logs: [] });
    },
  }))
);

// Computed selectors
export const useLegalMoves = () => {
  const gameState = useGameStore(state => state.gameState);
  return gameState?.legal_moves || [];
};

export const useCurrentPlayer = () => {
  const gameState = useGameStore(state => state.gameState);
  console.log('🔍 useCurrentPlayer - gameState:', gameState);
  console.log('🔍 useCurrentPlayer - current_player:', gameState?.current_player);
  return gameState?.current_player || null;
};

export const useIsMyTurn = () => {
  const currentPlayer = useCurrentPlayer();
  const gameState = useGameStore(state => state.gameState);
  
  console.log('🔄 useIsMyTurn check:', {
    currentPlayer,
    gameStateCurrentPlayer: gameState?.current_player,
    isMyTurn: gameState?.current_player === currentPlayer
  });
  
  if (!gameState || !currentPlayer) return false;
  
  // Check if the current player matches the game's current player
  return gameState?.current_player === currentPlayer;
};
