import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { API_BASE } from '../constants/gameConstants';

interface TrainingRun {
  id?: string;
  run_id: string;
  agent_id: string;
  algorithm: string;
  status: 'running' | 'completed' | 'stopped' | 'failed';
  start_time: string;
  end_time?: string;
  config?: Record<string, any>;
  agent_hyperparameters?: Record<string, any>;
  metadata?: {
    agent_config?: {
      path?: string;
      name?: string;
      config_short_name?: string;
      agent_id?: string;
      version?: number;
      sweep_variant?: string;
    };
    [key: string]: any;
  };
  metrics?: {
    episodes?: Array<{
      episode: number;
      total_reward: number;
      steps: number;
      win?: boolean;
      epsilon?: number;
    }>;
    rolling_win_rate?: Array<{
      episode: number;
      win_rate: number;
    }>;
  };
  checkpoint_paths?: Array<{
    episode: number;
    path: string;
  }>;
}

interface EvaluationRun {
  id?: string;
  training_run_id: string;
  checkpoint_path: string;
  opponent_type: string;
  games_played: number;
  win_rate: number;
  avg_reward: number;
  avg_game_length: number;
  created_at: string;
}

export const TrainingRunDetail: React.FC = () => {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const [run, setRun] = useState<TrainingRun | null>(null);
  const [evaluations, setEvaluations] = useState<EvaluationRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (runId) {
      fetchRun(runId);
      fetchEvaluations(runId);
    }
  }, [runId]);

  const fetchRun = async (id: string) => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE}/api/training-runs/${id}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch training run: ${response.statusText}`);
      }

      const data = await response.json();
      setRun(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load training run');
      console.error('Error fetching training run:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchEvaluations = async (id: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/training-runs/${id}/evaluations`);
      if (response.ok) {
        const data = await response.json();
        setEvaluations(data);
      }
    } catch (err) {
      console.error('Error fetching evaluations:', err);
    }
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleString();
  };

  const formatDuration = (startTime: string, endTime?: string): string => {
    if (!endTime) return 'Running...';

    const start = new Date(startTime);
    const end = new Date(endTime);
    const diffMs = end.getTime() - start.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffDays > 0) return `${diffDays}d ${diffHours % 24}h ${diffMins % 60}m`;
    if (diffHours > 0) return `${diffHours}h ${diffMins % 60}m`;
    return `${diffMins}m`;
  };

  const getStatusBadge = (status: string) => {
    const colors = {
      running: 'bg-blue-100 text-blue-800',
      completed: 'bg-green-100 text-green-800',
      stopped: 'bg-yellow-100 text-yellow-800',
      failed: 'bg-red-100 text-red-800',
    };

    return (
      <span className={`px-3 py-1 rounded-full text-sm font-semibold ${colors[status as keyof typeof colors] || 'bg-gray-100 text-gray-800'}`}>
        {status.toUpperCase()}
      </span>
    );
  };

  // Simple line chart component using SVG
  const LineChart: React.FC<{
    data: Array<{ episode: number; value: number }>;
    title: string;
    yLabel: string;
    color?: string;
  }> = ({ data, title, yLabel, color = '#3B82F6' }) => {
    if (!data || data.length === 0) {
      return (
        <div className="h-64 flex items-center justify-center text-gray-500">
          No data available
        </div>
      );
    }

    const width = 800;
    const height = 300;
    const padding = { top: 20, right: 20, bottom: 40, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    const maxEpisode = Math.max(...data.map(d => d.episode));
    const minEpisode = Math.min(...data.map(d => d.episode));
    const maxValue = Math.max(...data.map(d => d.value));
    const minValue = Math.min(...data.map(d => d.value));
    const valueRange = maxValue - minValue || 1;

    const scaleX = (episode: number) =>
      padding.left + ((episode - minEpisode) / (maxEpisode - minEpisode || 1)) * chartWidth;

    const scaleY = (value: number) =>
      padding.top + chartHeight - ((value - minValue) / valueRange) * chartHeight;

    const points = data.map(d => `${scaleX(d.episode)},${scaleY(d.value)}`).join(' ');

    return (
      <div className="bg-white rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-4">{title}</h3>
        <svg width={width} height={height} className="border border-gray-200 rounded">
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const y = padding.top + chartHeight - t * chartHeight;
            return (
              <line
                key={`grid-${t}`}
                x1={padding.left}
                y1={y}
                x2={width - padding.right}
                y2={y}
                stroke="#E5E7EB"
                strokeWidth={1}
              />
            );
          })}

          {/* Y-axis labels */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const value = minValue + t * valueRange;
            const y = padding.top + chartHeight - t * chartHeight;
            return (
              <text
                key={`y-label-${t}`}
                x={padding.left - 10}
                y={y + 4}
                textAnchor="end"
                fontSize="12"
                fill="#6B7280"
              >
                {value.toFixed(2)}
              </text>
            );
          })}

          {/* X-axis labels */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const episode = Math.round(minEpisode + t * (maxEpisode - minEpisode));
            const x = scaleX(episode);
            return (
              <text
                key={`x-label-${t}`}
                x={x}
                y={height - padding.bottom + 20}
                textAnchor="middle"
                fontSize="12"
                fill="#6B7280"
              >
                {episode}
              </text>
            );
          })}

          {/* Line */}
          <polyline
            points={points}
            fill="none"
            stroke={color}
            strokeWidth="2"
          />

          {/* Points */}
          {data.map((d, i) => (
            <circle
              key={i}
              cx={scaleX(d.episode)}
              cy={scaleY(d.value)}
              r="3"
              fill={color}
            />
          ))}

          {/* Labels */}
          <text
            x={width / 2}
            y={height - 10}
            textAnchor="middle"
            fontSize="12"
            fill="#6B7280"
          >
            Episode
          </text>
          <text
            x={20}
            y={height / 2}
            textAnchor="middle"
            fontSize="12"
            fill="#6B7280"
            transform={`rotate(-90, 20, ${height / 2})`}
          >
            {yLabel}
          </text>
        </svg>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <p className="mt-4 text-gray-600">Loading training run...</p>
        </div>
      </div>
    );
  }

  if (error || !run) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-lg p-8 max-w-md">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Error</h2>
          <p className="text-gray-600 mb-4">{error || 'Training run not found'}</p>
          <button
            onClick={() => navigate('/training')}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
          >
            Back to Training History
          </button>
        </div>
      </div>
    );
  }

  const episodes = run.metrics?.episodes || [];
  const rollingWinRate = run.metrics?.rolling_win_rate || [];

  // Calculate statistics
  const totalEpisodes = episodes.length;
  const avgReward = episodes.length > 0
    ? episodes.reduce((sum, e) => sum + e.total_reward, 0) / episodes.length
    : 0;
  const maxReward = episodes.length > 0
    ? Math.max(...episodes.map(e => e.total_reward))
    : 0;
  const finalWinRate = rollingWinRate.length > 0
    ? rollingWinRate[rollingWinRate.length - 1].win_rate
    : null;

  // Prepare chart data
  const rewardData = episodes.map(e => ({
    episode: e.episode,
    value: e.total_reward
  }));

  const winRateData = rollingWinRate.map(r => ({
    episode: r.episode,
    value: r.win_rate
  }));

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-800">Training Run Details</h1>
              <p className="text-sm text-gray-600 font-mono">{run.run_id}</p>
            </div>
            <button
              onClick={() => navigate('/training')}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
            >
              Back to History
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Run Info */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-500">Agent</p>
              <p className="text-lg font-semibold">{run.agent_id}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Algorithm</p>
              <p className="text-lg font-semibold">{run.algorithm}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Status</p>
              <div className="mt-1">{getStatusBadge(run.status)}</div>
            </div>
            <div>
              <p className="text-sm text-gray-500">Duration</p>
              <p className="text-lg font-semibold">{formatDuration(run.start_time, run.end_time)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Start Time</p>
              <p className="text-sm">{formatDate(run.start_time)}</p>
            </div>
            {run.end_time && (
              <div>
                <p className="text-sm text-gray-500">End Time</p>
                <p className="text-sm">{formatDate(run.end_time)}</p>
              </div>
            )}
          </div>
        </div>

        {/* Statistics Summary */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Statistics</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-blue-50 rounded-lg p-4">
              <p className="text-sm text-blue-600 font-medium">Total Episodes</p>
              <p className="text-2xl font-bold text-blue-900">{totalEpisodes}</p>
            </div>
            <div className="bg-green-50 rounded-lg p-4">
              <p className="text-sm text-green-600 font-medium">Avg Reward</p>
              <p className="text-2xl font-bold text-green-900">{avgReward.toFixed(2)}</p>
            </div>
            <div className="bg-purple-50 rounded-lg p-4">
              <p className="text-sm text-purple-600 font-medium">Max Reward</p>
              <p className="text-2xl font-bold text-purple-900">{maxReward.toFixed(2)}</p>
            </div>
            {finalWinRate !== null && (
              <div className="bg-yellow-50 rounded-lg p-4">
                <p className="text-sm text-yellow-600 font-medium">Final Win Rate</p>
                <p className="text-2xl font-bold text-yellow-900">{(finalWinRate * 100).toFixed(1)}%</p>
              </div>
            )}
          </div>
        </div>

        {/* Charts */}
        <div className="space-y-6 mb-6">
          <LineChart
            data={rewardData}
            title="Episode Reward"
            yLabel="Reward"
            color="#3B82F6"
          />

          {winRateData.length > 0 && (
            <LineChart
              data={winRateData}
              title="Rolling Win Rate"
              yLabel="Win Rate"
              color="#10B981"
            />
          )}
        </div>

        {/* Agent Config Summary */}
        {run.metadata?.agent_config && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Agent Configuration</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-500">Config Name</p>
                <p className="text-lg font-semibold">{run.metadata.agent_config.name || 'N/A'}</p>
              </div>
              {run.metadata.agent_config.config_short_name && (
                <div>
                  <p className="text-sm text-gray-500">Config ID</p>
                  <p className="text-lg font-semibold font-mono text-sm">{run.metadata.agent_config.config_short_name}</p>
                </div>
              )}
              {run.metadata.agent_config.version && (
                <div>
                  <p className="text-sm text-gray-500">Version</p>
                  <p className="text-lg font-semibold">v{run.metadata.agent_config.version}</p>
                </div>
              )}
              {run.metadata.agent_config.sweep_variant && (
                <div>
                  <p className="text-sm text-gray-500">Sweep Variant</p>
                  <p className="text-lg font-semibold">{run.metadata.agent_config.sweep_variant}</p>
                </div>
              )}
            </div>

            {/* Key Hyperparameters */}
            {run.agent_hyperparameters && (
              <div className="mt-6 pt-6 border-t border-gray-200">
                <h3 className="text-lg font-semibold mb-4">Key Hyperparameters</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {run.agent_hyperparameters.learning_rate && (
                    <div className="bg-blue-50 rounded-lg p-3">
                      <p className="text-xs text-blue-600 font-medium">Learning Rate</p>
                      <p className="text-lg font-bold text-blue-900">{run.agent_hyperparameters.learning_rate}</p>
                    </div>
                  )}
                  {run.agent_hyperparameters.gamma && (
                    <div className="bg-green-50 rounded-lg p-3">
                      <p className="text-xs text-green-600 font-medium">Gamma</p>
                      <p className="text-lg font-bold text-green-900">{run.agent_hyperparameters.gamma}</p>
                    </div>
                  )}
                  {run.agent_hyperparameters.batch_size && (
                    <div className="bg-purple-50 rounded-lg p-3">
                      <p className="text-xs text-purple-600 font-medium">Batch Size</p>
                      <p className="text-lg font-bold text-purple-900">{run.agent_hyperparameters.batch_size}</p>
                    </div>
                  )}
                  {run.agent_hyperparameters.n_steps && (
                    <div className="bg-yellow-50 rounded-lg p-3">
                      <p className="text-xs text-yellow-600 font-medium">N Steps</p>
                      <p className="text-lg font-bold text-yellow-900">{run.agent_hyperparameters.n_steps}</p>
                    </div>
                  )}
                </div>
                {run.agent_hyperparameters.network?.net_arch && (
                  <div className="mt-4">
                    <p className="text-sm text-gray-500 mb-2">Network Architecture</p>
                    <p className="text-sm font-mono bg-gray-50 rounded px-3 py-2">
                      {JSON.stringify(run.agent_hyperparameters.network.net_arch)}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Full Configuration */}
        {run.config && Object.keys(run.config).length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Full Training Configuration</h2>
            <div className="bg-gray-50 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-800">
                {JSON.stringify(run.config, null, 2)}
              </pre>
            </div>
          </div>
        )}

        {/* Evaluation Results */}
        {evaluations.length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Evaluation Results</h2>
            <div className="space-y-4">
              {evaluations.map((evalRun, idx) => (
                <div key={idx} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <p className="font-medium text-gray-900">
                        vs {evalRun.opponent_type.charAt(0).toUpperCase() + evalRun.opponent_type.slice(1)}
                      </p>
                      <p className="text-sm text-gray-600 mt-1">
                        {evalRun.games_played} games played
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-lg font-bold text-green-600">
                        {(evalRun.win_rate * 100).toFixed(1)}%
                      </p>
                      <p className="text-xs text-gray-500">Win Rate</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4 mt-3">
                    <div>
                      <p className="text-xs text-gray-500">Avg Reward</p>
                      <p className="text-sm font-semibold">{evalRun.avg_reward.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Avg Game Length</p>
                      <p className="text-sm font-semibold">{evalRun.avg_game_length.toFixed(1)} steps</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Evaluated</p>
                      <p className="text-sm font-semibold">
                        {new Date(evalRun.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Evaluation Command Hint */}
            {run.checkpoint_paths && run.checkpoint_paths.length > 0 && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <p className="text-sm font-medium text-gray-700 mb-2">Evaluate this model:</p>
                <div className="bg-gray-900 rounded-lg p-3 relative">
                  <code className="text-sm text-green-400 font-mono block break-all">
                    python training/evaluate_agent.py "{run.checkpoint_paths?.[run.checkpoint_paths.length - 1]?.path || ''}" --num-games 50
                  </code>
                  <button
                    onClick={() => {
                      const command = `python training/evaluate_agent.py "${run.checkpoint_paths?.[run.checkpoint_paths?.length - 1]?.path || ''}" --num-games 50`;
                      navigator.clipboard.writeText(command).then(() => {
                        alert('Command copied to clipboard!');
                      });
                    }}
                    className="absolute top-2 right-2 px-2 py-1 bg-gray-700 hover:bg-gray-600 text-white text-xs rounded transition"
                  >
                    Copy
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Run Metadata */}
        {run.metadata && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Run Metadata</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {run.metadata.git_hash && (
                <div>
                  <p className="text-sm text-gray-500">Git Hash</p>
                  <p className="text-sm font-mono text-gray-900">{run.metadata.git_hash.substring(0, 8)}...</p>
                </div>
              )}
              {run.metadata.git_branch && (
                <div>
                  <p className="text-sm text-gray-500">Git Branch</p>
                  <p className="text-sm font-semibold">{run.metadata.git_branch}</p>
                </div>
              )}
              {run.metadata.code_version && (
                <div>
                  <p className="text-sm text-gray-500">Code Version</p>
                  <p className="text-sm font-semibold">{run.metadata.code_version}</p>
                </div>
              )}
              {run.metadata.environment && (
                <div>
                  <p className="text-sm text-gray-500">Python Version</p>
                  <p className="text-sm font-semibold">{run.metadata.environment.python_version}</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Checkpoints */}
        {run.checkpoint_paths && run.checkpoint_paths.length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4">Checkpoints</h2>
            <div className="space-y-4">
              {run.checkpoint_paths.map((checkpoint, idx) => (
                <div key={idx} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <p className="font-medium text-gray-900">Episode {checkpoint.episode}</p>
                      <p className="text-sm text-gray-600 font-mono mt-1">{checkpoint.path}</p>
                    </div>
                  </div>

                  {/* Resume Command */}
                  <div className="mt-3">
                    <p className="text-sm font-medium text-gray-700 mb-2">Resume from this checkpoint:</p>
                    <div className="bg-gray-900 rounded-lg p-3 relative">
                      <code className="text-sm text-green-400 font-mono block break-all">
                        python training/trainer.py --resume-from-checkpoint "{checkpoint.path}"
                      </code>
                      <button
                        onClick={() => {
                          const command = `python training/trainer.py --resume-from-checkpoint "${checkpoint.path}"`;
                          navigator.clipboard.writeText(command).then(() => {
                            alert('Command copied to clipboard!');
                          });
                        }}
                        className="absolute top-2 right-2 px-2 py-1 bg-gray-700 hover:bg-gray-600 text-white text-xs rounded transition"
                      >
                        Copy
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

