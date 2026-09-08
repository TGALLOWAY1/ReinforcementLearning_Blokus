import React, { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useGameStore } from '../store/gameStore';
import {
  API_BASE,
  CHALLENGE_CHAMPION_BUDGET_MS,
  CHALLENGE_CHAMPION_PROFILE,
  DEPLOY_MCTS_PRESETS,
  IS_DEPLOY_PROFILE,
} from '../constants/gameConstants';
import {
  useArenaLeaderboard,
  pickDeployCurrentBestTrio,
  clampAgentConfigForDeploy,
  DEPLOY_TIME_BUDGET_CAP_MS,
  type LeaderboardAgent,
} from '../hooks/useArenaLeaderboard';
import { useChampion } from '../hooks/useChampion';
import {
  buildChampionGameConfig,
  championIsPlayable,
  type ChampionMetadata,
} from '../utils/championConfig';
import { ChampionCard } from './ChampionCard';
import { PentobiCard } from './PentobiCard';

interface GameConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  onGameCreated: () => void;
  /** When false (e.g. initial view), hide close button - user must start a game */
  canClose?: boolean;
}

const PLAYER_COLORS_ORDER = ['RED', 'BLUE', 'GREEN', 'YELLOW'] as const;

const buildDeployTierConfig = () => ({
  players: [
    { player: 'RED', agent_type: 'human', agent_config: {} },
    { player: 'BLUE', agent_type: 'mcts', agent_config: { difficulty: 'easy', time_budget_ms: DEPLOY_MCTS_PRESETS.easy } },
    { player: 'GREEN', agent_type: 'mcts', agent_config: { difficulty: 'medium', time_budget_ms: DEPLOY_MCTS_PRESETS.medium } },
    { player: 'YELLOW', agent_type: 'mcts', agent_config: { difficulty: 'hard', time_budget_ms: DEPLOY_MCTS_PRESETS.hard } },
  ],
  auto_start: true,
});

const buildChallengeChampionConfig = () => ({
  players: [
    { player: 'RED', agent_type: 'human', agent_config: {} },
    {
      player: 'BLUE',
      agent_type: 'mcts',
      agent_config: {
        profile: CHALLENGE_CHAMPION_PROFILE,
        time_budget_ms: CHALLENGE_CHAMPION_BUDGET_MS,
      },
    },
    {
      player: 'GREEN',
      agent_type: 'mcts',
      agent_config: {
        profile: CHALLENGE_CHAMPION_PROFILE,
        time_budget_ms: CHALLENGE_CHAMPION_BUDGET_MS,
      },
    },
    {
      player: 'YELLOW',
      agent_type: 'mcts',
      agent_config: {
        profile: CHALLENGE_CHAMPION_PROFILE,
        time_budget_ms: CHALLENGE_CHAMPION_BUDGET_MS,
      },
    },
  ],
  auto_start: true,
});

const formatMu = (mu: number) => mu.toFixed(1);

export const GameConfigModal: React.FC<GameConfigModalProps> = ({
  isOpen,
  onClose,
  onGameCreated,
  canClose = true,
}) => {
  const { createGame, connect, loadGame, setChampionMeta } = useGameStore();
  const championState = useChampion();
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [maxTime, setMaxTime] = useState<number>(DEPLOY_MCTS_PRESETS.hard);
  const [deployMode, setDeployMode] = useState<'tiers' | 'best'>('tiers');

  const leaderboard = useArenaLeaderboard();

  // --- Research-mode config state -------------------------------------------------
  const researchDefaultConfig = useMemo(
    () => ({
      players: [
        { player: 'RED', agent_type: 'human', agent_config: {} },
        { player: 'BLUE', agent_type: 'mcts', agent_config: { time_budget_ms: 1000 } },
        { player: 'GREEN', agent_type: 'mcts', agent_config: { time_budget_ms: 1000 } },
        { player: 'YELLOW', agent_type: 'mcts', agent_config: { time_budget_ms: 1000 } },
      ],
      auto_start: true,
    }),
    []
  );

  const [gameConfig, setGameConfig] = useState<any>(
    IS_DEPLOY_PROFILE ? buildDeployTierConfig() : researchDefaultConfig
  );
  // Track which leaderboard agent_id backs each research slot (null = custom/non-arena).
  const [slotArenaAgent, setSlotArenaAgent] = useState<Record<number, string | null>>({});
  const [expandedAdvanced, setExpandedAdvanced] = useState<number | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  // --- Shared helpers -------------------------------------------------------------

  const startFromConfig = async (config: any, champion: ChampionMetadata | null = null) => {
    setIsCreating(true);
    setError(null);
    // Tag (or clear) the champion banner for this game before it boots.
    setChampionMeta(champion);
    try {
      const gameId = await createGame(config);
      await connect(gameId);
      const storeState = useGameStore.getState();
      if (!storeState.gameState) {
        try {
          const response = await fetch(`${API_BASE}/api/games/${gameId}`);
          if (response.ok) {
            const gameState = await response.json();
            useGameStore.getState().setGameState(gameState);
          }
        } catch (err) {
          console.error('Failed to fetch game state via REST API:', err);
        }
      }
      onGameCreated();
      onClose();
    } catch (err) {
      console.error('Error creating game:', err);
      setError(err instanceof Error ? err.message : 'Failed to create game');
    } finally {
      setIsCreating(false);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsCreating(true);
    setError(null);

    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        const content = e.target?.result as string;
        const history = JSON.parse(content);
        if (!Array.isArray(history)) {
          throw new Error('Invalid save file format (expected array)');
        }
        await loadGame(history);
        onGameCreated();
        onClose();
      } catch (err) {
        console.error('Failed to load game:', err);
        setError(err instanceof Error ? err.message : 'Failed to load game file');
      } finally {
        setIsCreating(false);
        if (fileInputRef.current) fileInputRef.current.value = '';
      }
    };
    reader.onerror = () => {
      setError('Failed to read the file');
      setIsCreating(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    };
    reader.readAsText(file);
  };

  // --- Deploy mode ---------------------------------------------------------------

  const deployTrio = useMemo(
    () => pickDeployCurrentBestTrio(leaderboard.agents),
    [leaderboard.agents]
  );

  const buildDeployCurrentBestConfig = () => {
    if (!deployTrio) return null;
    const mkSlot = (
      player: 'BLUE' | 'GREEN' | 'YELLOW',
      difficulty: 'easy' | 'medium' | 'hard',
      source: LeaderboardAgent,
    ) => {
      const { config } = clampAgentConfigForDeploy(source.params);
      return {
        player,
        agent_type: 'mcts',
        agent_config: {
          ...config,
          difficulty, // required by deploy_validation.py
        },
      };
    };
    return {
      players: [
        { player: 'RED', agent_type: 'human', agent_config: {} },
        mkSlot('BLUE', 'easy', deployTrio.easy),
        mkSlot('GREEN', 'medium', deployTrio.medium),
        mkSlot('YELLOW', 'hard', deployTrio.hard),
      ],
      auto_start: true,
    };
  };

  const startDeployGame = () => {
    if (deployMode === 'best') {
      const config = buildDeployCurrentBestConfig();
      if (!config) {
        setError('No arena agents available — run scripts/arena.py first.');
        return;
      }
      return startFromConfig(config);
    }
    // Tiers mode: rescale by maxTime slider just like the original behavior did.
    return startFromConfig({
      players: [
        { player: 'RED', agent_type: 'human', agent_config: {} },
        { player: 'BLUE', agent_type: 'mcts', agent_config: { difficulty: 'easy', time_budget_ms: Math.floor(maxTime / 4.5) } },
        { player: 'GREEN', agent_type: 'mcts', agent_config: { difficulty: 'medium', time_budget_ms: Math.floor(maxTime / 2) } },
        { player: 'YELLOW', agent_type: 'mcts', agent_config: { difficulty: 'hard', time_budget_ms: maxTime } },
      ],
      auto_start: true,
    });
  };

  const startDeployArena = () => {
    // "Watch AI Battle" — all 4 seats are MCTS. Stamp easy/med/hard/pro labels
    // so deploy validation still accepts it (human required → fall back: just
    // use easy/med/hard + one extra distinct slot? No — deploy requires 1 human.
    // Keep arena a research-only feature for now by posting a 4-MCTS payload;
    // in deploy the backend will 400 because it requires 1 human. To keep the
    // button working in deploy we skip validation by only running it when a
    // human is included. Since this is a "watch" feature, we put the human at
    // seat RED but give them a random agent anyway is not allowed in deploy.
    // → Simplest: in deploy, arena just starts the 3-tier Human-vs-AI game too.
    return startFromConfig(buildDeployTierConfig());
  };

  const startChallengeChampion = () => startFromConfig(buildChallengeChampionConfig());

  const startPlayTheChampion = () => {
    const champion = championState.champion;
    if (!champion || !championIsPlayable(champion)) {
      setError(
        championState.error ||
          'No validated champion is available to play right now.',
      );
      return;
    }
    return startFromConfig(buildChampionGameConfig(champion), champion);
  };

  // --- Research mode -------------------------------------------------------------

  const arenaMctsAgents = useMemo(
    () => leaderboard.agents.filter((a) => a.type === 'mcts'),
    [leaderboard.agents]
  );
  const arenaNonMctsAgents = useMemo(
    () => leaderboard.agents.filter((a) => a.type !== 'mcts'),
    [leaderboard.agents]
  );

  const setPlayerField = (index: number, field: string, value: string) => {
    setGameConfig((prev: any) => ({
      ...prev,
      players: prev.players.map((player: any, i: number) =>
        i === index ? { ...player, [field]: value } : player
      ),
    }));
  };

  const addPlayer = () => {
    if (gameConfig.players.length < 4) {
      const usedColors = new Set(gameConfig.players.map((p: any) => p.player));
      const nextColor = PLAYER_COLORS_ORDER.find((c) => !usedColors.has(c)) ?? 'YELLOW';
      setGameConfig((prev: any) => ({
        ...prev,
        players: [
          ...prev.players,
          { player: nextColor, agent_type: 'random', agent_config: {} },
        ],
      }));
    }
  };

  const removePlayer = (index: number) => {
    if (gameConfig.players.length > 2) {
      setGameConfig((prev: any) => ({
        ...prev,
        players: prev.players.filter((_: any, i: number) => i !== index),
      }));
      setSlotArenaAgent((prev) => {
        const next: Record<number, string | null> = {};
        Object.entries(prev).forEach(([k, v]) => {
          const i = Number(k);
          if (i < index) next[i] = v;
          else if (i > index) next[i - 1] = v;
        });
        return next;
      });
    }
  };

  const updateAdvancedConfig = (playerIndex: number, key: string, value: any) => {
    setGameConfig((prev: any) => ({
      ...prev,
      players: prev.players.map((p: any, i: number) =>
        i === playerIndex
          ? { ...p, agent_config: { ...p.agent_config, [key]: value } }
          : p
      ),
    }));
    // Any manual edit means the slot no longer matches its source arena agent verbatim.
    setSlotArenaAgent((prev) => ({ ...prev, [playerIndex]: null }));
  };

  const handleAgentTypeChange = (index: number, nextType: string) => {
    setGameConfig((prev: any) => ({
      ...prev,
      players: prev.players.map((p: any, i: number) => {
        if (i !== index) return p;
        if (nextType === 'mcts') {
          return { ...p, agent_type: 'mcts', agent_config: { time_budget_ms: 1000 } };
        }
        return { ...p, agent_type: nextType, agent_config: {} };
      }),
    }));
    setSlotArenaAgent((prev) => ({ ...prev, [index]: null }));
  };

  const applyArenaAgentToSlot = (index: number, agentId: string) => {
    if (!agentId) {
      setSlotArenaAgent((prev) => ({ ...prev, [index]: null }));
      return;
    }
    const agent = leaderboard.agents.find((a) => a.agent_id === agentId);
    if (!agent) return;
    setGameConfig((prev: any) => ({
      ...prev,
      players: prev.players.map((p: any, i: number) =>
        i === index
          ? { ...p, agent_type: agent.type, agent_config: { ...agent.params } }
          : p
      ),
    }));
    setSlotArenaAgent((prev) => ({ ...prev, [index]: agentId }));
  };

  const handleCreateGame = () => startFromConfig(gameConfig);

  if (!isOpen) return null;

  // ================================================================================
  // DEPLOY PROFILE
  // ================================================================================
  if (IS_DEPLOY_PROFILE) {
    const hasArena = deployTrio !== null;
    const trio = deployTrio;

    return (
      <div className="fixed inset-0 bg-charcoal-900 flex items-center justify-center z-50 p-4">
        <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg max-w-md w-full p-8 text-center relative">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="absolute top-4 left-4 text-gray-400 hover:text-neon-blue transition-colors"
            title="Advanced Settings"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
          {canClose && (
            <button
              onClick={onClose}
              className="absolute top-4 right-4 text-gray-400 hover:text-gray-200 transition-colors"
              aria-label="Close"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
          <h1 className="text-3xl font-bold text-gray-200 mb-2">Blokus</h1>

          <div className="mb-4 flex justify-center">
            <Link
              to="/tour"
              className="inline-flex items-center gap-2 rounded-full border border-neon-cyan/40 bg-neon-cyan/5 px-4 py-1.5 text-xs font-semibold text-neon-cyan transition-colors hover:border-neon-cyan hover:bg-neon-cyan/10"
              title="New here? Take the interactive tour"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
              </svg>
              Take the Tour
            </Link>
          </div>

          {showSettings && deployMode === 'tiers' && (
            <div className="mb-6 bg-charcoal-900 p-4 rounded-lg border border-charcoal-700 text-left">
              <label className="block text-sm text-gray-300 font-medium mb-2">
                Max AI Thinking Time: {maxTime / 1000}s
              </label>
              <input
                type="range"
                min="400"
                max={DEPLOY_TIME_BUDGET_CAP_MS}
                step="100"
                value={maxTime}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setMaxTime(parseInt(e.target.value))}
                className="w-full accent-neon-blue h-2 bg-charcoal-700 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-gray-500 mt-2">
                Adjusts the maximum time the Hard AI will think. Easy and Medium scale proportionally.
              </p>
            </div>
          )}

          {error && (
            <div className="bg-red-900 border border-red-700 rounded-lg p-4 mb-6 text-red-200 text-sm">
              {error}
            </div>
          )}

          {/* Headline demo: Play the Champion (registry-backed) */}
          <div className="mb-4">
            <ChampionCard
              state={championState}
              isCreating={isCreating}
              onPlay={startPlayTheChampion}
            />
          </div>

          {/* Human protocol: natively served Pentobi (M4/M5) */}
          <div className="mb-4">
            <PentobiCard isCreating={isCreating} onPlay={(config) => startFromConfig(config)} />
          </div>

          {/* Mode tabs */}
          <div className="flex mb-4 rounded-lg overflow-hidden border border-charcoal-700">
            <button
              onClick={() => setDeployMode('tiers')}
              className={`flex-1 py-2 text-xs font-semibold uppercase tracking-wider transition-colors ${
                deployMode === 'tiers'
                  ? 'bg-neon-blue text-black'
                  : 'bg-charcoal-900 text-gray-400 hover:text-gray-200'
              }`}
            >
              Quick Tiers
            </button>
            <button
              onClick={() => setDeployMode('best')}
              disabled={!hasArena && !leaderboard.isLoading}
              title={
                !hasArena && !leaderboard.isLoading
                  ? 'No arena runs yet — run scripts/arena.py to populate the leaderboard.'
                  : undefined
              }
              className={`flex-1 py-2 text-xs font-semibold uppercase tracking-wider transition-colors ${
                deployMode === 'best'
                  ? 'bg-neon-blue text-black'
                  : 'bg-charcoal-900 text-gray-400 hover:text-gray-200 disabled:opacity-40 disabled:cursor-not-allowed'
              }`}
            >
              Current Best
            </button>
          </div>

          {/* Mode body */}
          <div className="flex flex-col gap-3 mb-4">
            {deployMode === 'tiers' ? (
              <div className="text-left">
                <p className="text-xs text-gray-500 uppercase tracking-wider mb-2 font-bold">Play vs AI</p>
                <div className="space-y-2">
                  <button
                    onClick={startDeployGame}
                    disabled={isCreating}
                    className={`w-full py-3 px-6 rounded-lg font-medium transition-colors duration-200 text-left flex items-center justify-between ${
                      isCreating
                        ? 'bg-gray-600 cursor-not-allowed text-gray-400'
                        : 'bg-neon-blue hover:bg-neon-blue/80 text-black'
                    }`}
                  >
                    <span>{isCreating ? 'Starting...' : 'Start Game'}</span>
                    <span className="text-xs opacity-70">You (Red) vs Easy · Med · Hard</span>
                  </button>
                  <button
                    onClick={startChallengeChampion}
                    disabled={isCreating}
                    className={`w-full py-3 px-6 rounded-lg font-medium border transition-colors duration-200 text-left flex items-center justify-between ${
                      isCreating
                        ? 'border-charcoal-600 bg-charcoal-700 text-gray-500 cursor-not-allowed'
                        : 'border-neon-green bg-neon-green/10 hover:bg-neon-green/20 text-neon-green'
                    }`}
                  >
                    <span>{isCreating ? 'Starting...' : 'Challenge Champion'}</span>
                    <span className="text-xs opacity-70">Adaptive full-stack MCTS trio</span>
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-left">
                <p className="text-xs text-gray-500 uppercase tracking-wider mb-2 font-bold">
                  Play vs Current Best
                </p>
                {leaderboard.isLoading ? (
                  <div className="text-xs text-gray-500 py-4 text-center">Loading leaderboard…</div>
                ) : leaderboard.error ? (
                  <div className="text-xs text-red-400 py-4">{leaderboard.error}</div>
                ) : !trio ? (
                  <div className="text-xs text-gray-500 py-4">
                    No arena runs yet. Run <code>scripts/arena.py</code> to populate the leaderboard.
                  </div>
                ) : (
                  <>
                    <div className="mb-3 rounded-lg border border-charcoal-700 bg-charcoal-900 p-3 text-xs">
                      {[
                        { label: 'Easy', agent: trio.easy, color: 'text-gray-400' },
                        { label: 'Medium', agent: trio.medium, color: 'text-gray-200' },
                        { label: 'Hard', agent: trio.hard, color: 'text-neon-blue' },
                      ].map(({ label, agent, color }) => (
                        <div key={label} className="flex items-center justify-between py-0.5">
                          <span className={`font-semibold ${color}`}>{label}</span>
                          <span className="text-gray-300 font-mono truncate ml-3">{agent.agent_id}</span>
                          <span className="text-gray-500 ml-2">μ {formatMu(agent.mu)} · #{agent.rank}</span>
                        </div>
                      ))}
                    </div>
                    <button
                      onClick={startDeployGame}
                      disabled={isCreating}
                      className={`w-full py-3 px-6 rounded-lg font-medium transition-colors duration-200 text-left flex items-center justify-between ${
                        isCreating
                          ? 'bg-gray-600 cursor-not-allowed text-gray-400'
                          : 'bg-neon-blue hover:bg-neon-blue/80 text-black'
                      }`}
                    >
                      <span>{isCreating ? 'Starting...' : 'Start Game'}</span>
                      <span className="text-xs opacity-70">You (Red) vs reigning agents</span>
                    </button>
                  </>
                )}
              </div>
            )}

            {/* Watch AI Battle (tiers only) */}
            {deployMode === 'tiers' && (
              <div className="text-left">
                <p className="text-xs text-gray-500 uppercase tracking-wider mb-2 font-bold">Watch AI Battle</p>
                <button
                  onClick={startDeployArena}
                  disabled={isCreating}
                  className={`w-full py-3 px-6 rounded-lg font-medium border transition-colors duration-200 text-left flex items-center justify-between ${
                    isCreating
                      ? 'border-charcoal-600 bg-charcoal-700 text-gray-500 cursor-not-allowed'
                      : 'border-neon-blue bg-neon-blue/5 hover:bg-neon-blue/10 text-neon-blue'
                  }`}
                >
                  <span className="flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    {isCreating ? 'Starting...' : 'MCTS Arena'}
                  </span>
                  <span className="text-xs opacity-70">Easy · Med · Hard</span>
                </button>
              </div>
            )}
          </div>

          {/* Load game */}
          <input
            type="file"
            accept=".json"
            ref={fileInputRef}
            onChange={handleFileUpload}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isCreating}
            className={`w-full py-2 px-6 rounded-lg font-medium border transition-colors duration-200 text-sm ${
              isCreating
                ? 'bg-charcoal-700 border-charcoal-600 text-gray-500 cursor-not-allowed'
                : 'bg-charcoal-800 border-charcoal-600 text-gray-400 hover:bg-charcoal-700 hover:text-white'
            }`}
          >
            Load Saved Game
          </button>
        </div>
      </div>
    );
  }

  // ================================================================================
  // RESEARCH PROFILE
  // ================================================================================
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-charcoal-800 border border-charcoal-700 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-200">Game Configuration</h2>
            <div className="flex items-center gap-3">
              <Link
                to="/tour"
                className="inline-flex items-center gap-1.5 rounded-full border border-neon-cyan/40 bg-neon-cyan/5 px-3 py-1.5 text-xs font-semibold text-neon-cyan transition-colors hover:border-neon-cyan hover:bg-neon-cyan/10"
                title="New here? Take the interactive tour"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
                Take the Tour
              </Link>
              {canClose && (
                <button
                  onClick={onClose}
                  className="text-gray-400 hover:text-gray-200 transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              )}
            </div>
          </div>

          {error && (
            <div className="bg-red-900 border border-red-700 rounded-lg p-4 mb-6">
              <div className="text-red-200">{error}</div>
            </div>
          )}

          {/* Arena leaderboard banner */}
          <div className="mb-6 p-4 bg-charcoal-900 border border-charcoal-700 rounded-lg text-sm">
            <div className="flex items-center justify-between mb-1">
              <span className="text-gray-200 font-semibold">Arena Leaderboard</span>
              <span className="text-xs text-gray-500">
                {leaderboard.isLoading
                  ? 'loading…'
                  : leaderboard.error
                  ? 'error'
                  : leaderboard.runId
                  ? `run ${leaderboard.runId}`
                  : 'no runs yet'}
              </span>
            </div>
            {leaderboard.error ? (
              <div className="text-xs text-red-400">{leaderboard.error}</div>
            ) : leaderboard.agents.length === 0 && !leaderboard.isLoading ? (
              <div className="text-xs text-gray-500">
                Run <code>python scripts/arena.py</code> to populate rated agents.
              </div>
            ) : (
              <div className="text-xs text-gray-400">
                {leaderboard.agents.length} rated agents available as opponents. Pick any per slot below.
              </div>
            )}
          </div>

          {/* Registry-backed "Play the Champion" demo */}
          <div className="mb-6">
            <ChampionCard
              state={championState}
              isCreating={isCreating}
              onPlay={startPlayTheChampion}
            />
          </div>

          {/* Human protocol: natively served Pentobi (M4/M5) */}
          <div className="mb-6">
            <PentobiCard isCreating={isCreating} onPlay={(config) => startFromConfig(config)} />
          </div>

          <div className="mb-6 rounded-lg border border-neon-green/30 bg-neon-green/5 p-4">
            <div className="flex items-center justify-between gap-4">
              <div>
                <div className="text-sm font-semibold text-neon-green">Challenge Champion</div>
                <div className="text-xs text-gray-400">
                  Start immediately against the adaptive human-play profile used in challenge mode.
                </div>
              </div>
              <button
                onClick={startChallengeChampion}
                disabled={isCreating}
                className={`shrink-0 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                  isCreating
                    ? 'bg-charcoal-700 text-gray-500 cursor-not-allowed'
                    : 'bg-neon-green text-black hover:bg-neon-green/80'
                }`}
              >
                {isCreating ? 'Starting...' : 'Start Challenge'}
              </button>
            </div>
          </div>

          <div className="border-t border-charcoal-700 pt-2">
            <h3 className="text-lg font-semibold text-gray-200 mb-4">Players ({gameConfig.players.length}/4)</h3>

            <div className="space-y-3 mb-4">
              {gameConfig.players.map((player: any, index: number) => {
                const currentArenaId = slotArenaAgent[index] ?? '';
                return (
                  <div key={index} className="space-y-2">
                    <div className="flex items-center space-x-3">
                      <div className="w-24">
                        <select
                          value={player.player}
                          onChange={(e) => setPlayerField(index, 'player', e.target.value)}
                          className="w-full bg-charcoal-900 border border-charcoal-700 text-gray-200 rounded-md px-3 py-2 text-sm focus:outline-none focus:border-neon-blue"
                        >
                          <option value="RED">Red</option>
                          <option value="BLUE">Blue</option>
                          <option value="GREEN">Green</option>
                          <option value="YELLOW">Yellow</option>
                        </select>
                      </div>
                      <div className="flex-1">
                        <select
                          value={player.agent_type}
                          onChange={(e) => handleAgentTypeChange(index, e.target.value)}
                          className="w-full bg-charcoal-900 border border-charcoal-700 text-gray-200 rounded-md px-3 py-2 text-sm focus:outline-none focus:border-neon-blue"
                        >
                          <option value="human">Human</option>
                          <option value="random">Random Agent</option>
                          <option value="heuristic">Heuristic Agent</option>
                          <option value="mcts">MCTS Agent</option>
                        </select>
                      </div>
                      {player.agent_type === 'mcts' && (
                        <button
                          onClick={() =>
                            setExpandedAdvanced(expandedAdvanced === index ? null : index)
                          }
                          className={`text-xs px-2 py-1 rounded border transition-colors ${
                            expandedAdvanced === index
                              ? 'border-neon-blue text-neon-blue bg-neon-blue/10'
                              : 'border-charcoal-600 text-gray-400 hover:text-neon-blue hover:border-neon-blue'
                          }`}
                          title="Edit MCTS parameters"
                        >
                          Tune
                        </button>
                      )}
                      {gameConfig.players.length > 2 && (
                        <button
                          onClick={() => removePlayer(index)}
                          className="text-red-400 hover:text-red-300 text-sm px-2"
                        >
                          Remove
                        </button>
                      )}
                    </div>

                    {/* Arena agent picker (per slot, when agent is MCTS or any rated type) */}
                    {(player.agent_type === 'mcts' ||
                      player.agent_type === 'random' ||
                      player.agent_type === 'heuristic') &&
                      leaderboard.agents.length > 0 && (
                        <div className="ml-[6.5rem] pl-0">
                          <label className="text-[10px] uppercase tracking-wider text-gray-500 font-semibold">
                            From arena leaderboard
                          </label>
                          <select
                            value={currentArenaId}
                            onChange={(e) => applyArenaAgentToSlot(index, e.target.value)}
                            className="mt-1 w-full bg-charcoal-900 border border-charcoal-700 text-gray-200 rounded-md px-3 py-2 text-xs focus:outline-none focus:border-neon-blue"
                          >
                            <option value="">— Custom / default —</option>
                            {player.agent_type === 'mcts' && arenaMctsAgents.length > 0 && (
                              <optgroup label="MCTS agents">
                                {arenaMctsAgents.map((a) => (
                                  <option key={a.agent_id} value={a.agent_id}>
                                    {a.agent_id} — μ {formatMu(a.mu)} (rank {a.rank})
                                  </option>
                                ))}
                              </optgroup>
                            )}
                            {arenaNonMctsAgents.length > 0 && (
                              <optgroup label="Baselines">
                                {arenaNonMctsAgents.map((a) => (
                                  <option key={a.agent_id} value={a.agent_id}>
                                    {a.agent_id} ({a.type}) — μ {formatMu(a.mu)} (rank {a.rank})
                                  </option>
                                ))}
                              </optgroup>
                            )}
                          </select>
                        </div>
                      )}

                    {/* Advanced MCTS Configuration */}
                    {player.agent_type === 'mcts' && expandedAdvanced === index && (
                      <div className="ml-4 p-3 bg-charcoal-900 border border-charcoal-700 rounded-lg text-xs space-y-3">
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="block text-gray-500 mb-0.5">Time Budget (ms)</label>
                            <input
                              type="number"
                              min={50}
                              max={10000}
                              step={50}
                              value={player.agent_config?.time_budget_ms ?? 1000}
                              onChange={(e) =>
                                updateAdvancedConfig(
                                  index,
                                  'time_budget_ms',
                                  parseInt(e.target.value) || 1000
                                )
                              }
                              className="w-full bg-charcoal-800 border border-charcoal-600 text-gray-200 rounded px-2 py-1 text-xs"
                            />
                          </div>
                          <div>
                            <label className="block text-gray-500 mb-0.5">Exploration C</label>
                            <input
                              type="number"
                              min={0}
                              max={5}
                              step={0.1}
                              value={player.agent_config?.exploration_constant ?? 1.414}
                              onChange={(e) =>
                                updateAdvancedConfig(
                                  index,
                                  'exploration_constant',
                                  parseFloat(e.target.value) || 1.414
                                )
                              }
                              className="w-full bg-charcoal-800 border border-charcoal-600 text-gray-200 rounded px-2 py-1 text-xs"
                            />
                          </div>
                        </div>

                        <div>
                          <div className="text-gray-400 font-semibold uppercase tracking-wider mb-1" style={{ fontSize: '10px' }}>
                            L3: Action Reduction
                          </div>
                          <div className="flex gap-4">
                            <label className="flex items-center gap-1 text-gray-300">
                              <input
                                type="checkbox"
                                checked={!!player.agent_config?.progressive_widening_enabled}
                                onChange={(e) => updateAdvancedConfig(index, 'progressive_widening_enabled', e.target.checked)}
                                className="accent-neon-blue"
                              />
                              Prog. Widening
                            </label>
                            <label className="flex items-center gap-1 text-gray-300">
                              <input
                                type="checkbox"
                                checked={!!player.agent_config?.progressive_history_enabled}
                                onChange={(e) => updateAdvancedConfig(index, 'progressive_history_enabled', e.target.checked)}
                                className="accent-neon-blue"
                              />
                              Prog. History
                            </label>
                          </div>
                        </div>

                        <div>
                          <div className="text-gray-400 font-semibold uppercase tracking-wider mb-1" style={{ fontSize: '10px' }}>
                            L4: Simulation Strategy
                          </div>
                          <div className="grid grid-cols-3 gap-2">
                            <div>
                              <label className="block text-gray-500 mb-0.5">Rollout Policy</label>
                              <select
                                value={player.agent_config?.rollout_policy ?? 'heuristic'}
                                onChange={(e) => updateAdvancedConfig(index, 'rollout_policy', e.target.value)}
                                className="w-full bg-charcoal-800 border border-charcoal-600 text-gray-200 rounded px-2 py-1 text-xs"
                              >
                                <option value="random">Random</option>
                                <option value="heuristic">Heuristic</option>
                                <option value="two_ply">Two-Ply</option>
                              </select>
                            </div>
                            <div>
                              <label className="block text-gray-500 mb-0.5">Cutoff Depth</label>
                              <input
                                type="number"
                                min={0}
                                max={100}
                                step={5}
                                value={player.agent_config?.rollout_cutoff_depth ?? ''}
                                placeholder="None"
                                onChange={(e) =>
                                  updateAdvancedConfig(
                                    index,
                                    'rollout_cutoff_depth',
                                    e.target.value ? parseInt(e.target.value) : null
                                  )
                                }
                                className="w-full bg-charcoal-800 border border-charcoal-600 text-gray-200 rounded px-2 py-1 text-xs"
                              />
                            </div>
                            <div>
                              <label className="block text-gray-500 mb-0.5">Minimax Alpha</label>
                              <input
                                type="number"
                                min={0}
                                max={1}
                                step={0.05}
                                value={player.agent_config?.minimax_backup_alpha ?? 0}
                                onChange={(e) =>
                                  updateAdvancedConfig(index, 'minimax_backup_alpha', parseFloat(e.target.value) || 0)
                                }
                                className="w-full bg-charcoal-800 border border-charcoal-600 text-gray-200 rounded px-2 py-1 text-xs"
                              />
                            </div>
                          </div>
                        </div>

                        <div>
                          <div className="text-gray-400 font-semibold uppercase tracking-wider mb-1" style={{ fontSize: '10px' }}>
                            L5: RAVE & History
                          </div>
                          <div className="flex gap-4 items-end">
                            <label className="flex items-center gap-1 text-gray-300">
                              <input
                                type="checkbox"
                                checked={!!player.agent_config?.rave_enabled}
                                onChange={(e) => updateAdvancedConfig(index, 'rave_enabled', e.target.checked)}
                                className="accent-neon-blue"
                              />
                              RAVE
                            </label>
                            {player.agent_config?.rave_enabled && (
                              <div>
                                <label className="block text-gray-500 mb-0.5">RAVE k</label>
                                <input
                                  type="number"
                                  min={100}
                                  max={10000}
                                  step={100}
                                  value={player.agent_config?.rave_k ?? 1000}
                                  onChange={(e) => updateAdvancedConfig(index, 'rave_k', parseFloat(e.target.value) || 1000)}
                                  className="w-24 bg-charcoal-800 border border-charcoal-600 text-gray-200 rounded px-2 py-1 text-xs"
                                />
                              </div>
                            )}
                            <label className="flex items-center gap-1 text-gray-300">
                              <input
                                type="checkbox"
                                checked={!!player.agent_config?.nst_enabled}
                                onChange={(e) => updateAdvancedConfig(index, 'nst_enabled', e.target.checked)}
                                className="accent-neon-blue"
                              />
                              NST
                            </label>
                          </div>
                        </div>

                        <div>
                          <div className="text-gray-400 font-semibold uppercase tracking-wider mb-1" style={{ fontSize: '10px' }}>
                            L7: Opponent Modeling
                          </div>
                          <div className="flex flex-wrap gap-3">
                            <label className="flex items-center gap-1 text-gray-300">
                              <input
                                type="checkbox"
                                checked={!!player.agent_config?.opponent_modeling_enabled}
                                onChange={(e) => updateAdvancedConfig(index, 'opponent_modeling_enabled', e.target.checked)}
                                className="accent-neon-blue"
                              />
                              Enabled
                            </label>
                            <label className="flex items-center gap-1 text-gray-300">
                              <input
                                type="checkbox"
                                checked={!!player.agent_config?.alliance_detection_enabled}
                                onChange={(e) => updateAdvancedConfig(index, 'alliance_detection_enabled', e.target.checked)}
                                className="accent-neon-blue"
                              />
                              Alliances
                            </label>
                            <label className="flex items-center gap-1 text-gray-300">
                              <input
                                type="checkbox"
                                checked={!!player.agent_config?.kingmaker_detection_enabled}
                                onChange={(e) => updateAdvancedConfig(index, 'kingmaker_detection_enabled', e.target.checked)}
                                className="accent-neon-blue"
                              />
                              King-maker
                            </label>
                          </div>
                        </div>

                        <div>
                          <div className="text-gray-400 font-semibold uppercase tracking-wider mb-1" style={{ fontSize: '10px' }}>
                            L9: Meta-Optimization
                          </div>
                          <div className="flex flex-wrap gap-3">
                            <label className="flex items-center gap-1 text-gray-300">
                              <input
                                type="checkbox"
                                checked={!!player.agent_config?.adaptive_exploration_enabled}
                                onChange={(e) => updateAdvancedConfig(index, 'adaptive_exploration_enabled', e.target.checked)}
                                className="accent-neon-blue"
                              />
                              Adaptive C
                            </label>
                            <label className="flex items-center gap-1 text-gray-300">
                              <input
                                type="checkbox"
                                checked={!!player.agent_config?.sufficiency_threshold_enabled}
                                onChange={(e) => updateAdvancedConfig(index, 'sufficiency_threshold_enabled', e.target.checked)}
                                className="accent-neon-blue"
                              />
                              Sufficiency
                            </label>
                            <label className="flex items-center gap-1 text-gray-300">
                              <input
                                type="checkbox"
                                checked={!!player.agent_config?.loss_avoidance_enabled}
                                onChange={(e) => updateAdvancedConfig(index, 'loss_avoidance_enabled', e.target.checked)}
                                className="accent-neon-blue"
                              />
                              Loss Avoidance
                            </label>
                          </div>
                        </div>

                        <div>
                          <label className="flex items-center gap-1 text-gray-300">
                            <input
                              type="checkbox"
                              checked={!!player.agent_config?.enable_search_trace}
                              onChange={(e) => updateAdvancedConfig(index, 'enable_search_trace', e.target.checked)}
                              className="accent-neon-blue"
                            />
                            Enable Search Trace (for MCTS visualization)
                          </label>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {gameConfig.players.length < 4 && (
              <button
                onClick={addPlayer}
                className="mt-2 text-neon-blue hover:text-neon-blue/80 text-sm"
              >
                + Add Player
              </button>
            )}

            <div className="flex items-center my-6">
              <input
                type="checkbox"
                id="auto_start"
                checked={gameConfig.auto_start}
                onChange={(e) =>
                  setGameConfig((prev: any) => ({ ...prev, auto_start: e.target.checked }))
                }
                className="mr-2"
              />
              <label htmlFor="auto_start" className="text-sm text-gray-300">
                Auto-start game
              </label>
            </div>

            <div className="flex space-x-3">
              <button
                onClick={handleCreateGame}
                disabled={isCreating}
                className={`flex-1 py-3 px-6 rounded-lg font-medium text-white transition-colors duration-200 ${
                  isCreating ? 'bg-gray-600 cursor-not-allowed' : 'bg-neon-blue hover:bg-neon-blue/80'
                }`}
              >
                {isCreating ? 'Creating Game...' : 'Start New Game'}
              </button>

              <input
                type="file"
                accept=".json"
                ref={fileInputRef}
                onChange={handleFileUpload}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isCreating}
                className={`py-3 px-6 rounded-lg font-medium border transition-colors duration-200 ${
                  isCreating
                    ? 'bg-charcoal-700 border-charcoal-600 text-gray-500 cursor-not-allowed'
                    : 'bg-charcoal-800 border-charcoal-600 text-gray-300 hover:bg-charcoal-700 hover:text-white'
                }`}
              >
                Load From File
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
