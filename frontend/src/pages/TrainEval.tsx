import React from 'react';
import { useNavigate } from 'react-router-dom';

export const TrainEval: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-800">Training & Evaluation</h1>
              <p className="text-sm text-gray-600">
                Monitor agent performance and training progress
              </p>
            </div>
            <button
              onClick={() => navigate('/training')}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
            >
              View Training History
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Training Progress */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Training Progress
            </h2>
            <div className="space-y-4">
              <div className="text-center py-12 text-gray-500">
                <div className="text-4xl mb-4">📊</div>
                <div>Training charts will appear here</div>
                <div className="text-sm mt-2">
                  Run evaluation scripts to see performance metrics
                </div>
              </div>
            </div>
          </div>

          {/* Agent Comparison */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Agent Performance
            </h2>
            <div className="space-y-4">
              <div className="text-center py-12 text-gray-500">
                <div className="text-4xl mb-4">🏆</div>
                <div>Agent comparison charts will appear here</div>
                <div className="text-sm mt-2">
                  Win rates, average scores, and other metrics
                </div>
              </div>
            </div>
          </div>

          {/* Recent Games */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Recent Games
            </h2>
            <div className="space-y-4">
              <div className="text-center py-12 text-gray-500">
                <div className="text-4xl mb-4">🎮</div>
                <div>Game history will appear here</div>
                <div className="text-sm mt-2">
                  Track game outcomes and agent performance
                </div>
              </div>
            </div>
          </div>

          {/* Evaluation Results */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Evaluation Results
            </h2>
            <div className="space-y-4">
              <div className="text-center py-12 text-gray-500">
                <div className="text-4xl mb-4">📈</div>
                <div>Evaluation results will appear here</div>
                <div className="text-sm mt-2">
                  Detailed performance analysis and statistics
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Instructions */}
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-800 mb-3">
            How to Use This Page
          </h3>
          <div className="text-blue-700 space-y-2">
            <div>• Run the arena script to generate evaluation data</div>
            <div>• Use training scripts to monitor learning progress</div>
            <div>• Charts and metrics will automatically populate</div>
            <div>• Compare different agent configurations</div>
          </div>
        </div>
      </div>
    </div>
  );
};
