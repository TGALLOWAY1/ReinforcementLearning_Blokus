import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Play } from '../pages/Play';
import { useGameStore } from '../store/gameStore';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';

// Mock data generator
const mockGameState = (currentPlayer: string, agentType: string) => ({
    game_id: 'test-game',
    status: 'in_progress',
    current_player: currentPlayer,
    board: Array(20).fill(Array(20).fill(0)),
    scores: { RED: 0, BLUE: 0, GREEN: 0, YELLOW: 0 },
    pieces_used: { RED: [], BLUE: [], GREEN: [], YELLOW: [] },
    move_count: 0,
    game_over: false,
    winner: null,
    legal_moves: [],
    players: [
        { player: 'RED', agent_type: 'human' },
        { player: 'BLUE', agent_type: agentType },
    ],
});

// Mock the store
const basicState = mockGameState('RED', 'mcts');
const mockStore = {
    ...basicState,
    gameState: basicState,
    connectionStatus: 'connected',
    activeRightTab: 'main',
    boardOverlay: null,
    isPaused: false,
    error: null,
    selectedPiece: null,
    pieceOrientation: 0,
    setError: vi.fn(),
    selectPiece: vi.fn(),
    setPieceOrientation: vi.fn(),
    togglePause: vi.fn(),
    makeMove: vi.fn(),
    passTurn: vi.fn(),
    saveGame: vi.fn(),
};

vi.mock('../store/gameStore', () => ({
    useGameStore: Object.assign(
        (selector?: (s: any) => any) => selector ? selector(mockStore) : mockStore,
        {
            getState: () => mockStore,
            setState: vi.fn(),
            subscribe: vi.fn(),
        }
    ),
}));



describe('Header Persistence', () => {
    it('should keep critical UI elements visible during an AI turn', () => {
        // 1. Initial state is human turn (RED)
        // mockStore is already initialized for RED human

        const { rerender } = render(
            <BrowserRouter>
                <Play />
            </BrowserRouter>
        );

        // Verify hint button exists
        expect(screen.getByTestId('critical-ui-hint')).toBeInTheDocument();
        expect(screen.getByTestId('critical-ui-pass')).toBeInTheDocument();
        expect(screen.getByTestId('critical-ui-save')).toBeInTheDocument();

        // 2. Switch to AI turn (BLUE)
        Object.assign(mockStore, mockGameState('BLUE', 'mcts'));
        mockStore.gameState = { ...mockStore }; // Keep gameState in sync if needed

        rerender(
            <BrowserRouter>
                <Play />
            </BrowserRouter>
        );

        // CRITICAL CHECK: Hint button SHOULD STILL EXIST even if current_player is AI
        // because there is at least one human player in the game.
        expect(screen.getByTestId('critical-ui-hint')).toBeInTheDocument();
        expect(screen.getByTestId('critical-ui-pass')).toBeInTheDocument();
        expect(screen.getByTestId('critical-ui-save')).toBeInTheDocument();
    });
});
