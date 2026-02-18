import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Home } from './pages/Home';
import { Play } from './pages/Play';
import { TrainEval } from './pages/TrainEval';
import { TrainingHistory } from './pages/TrainingHistory';
import { TrainingRunDetail } from './pages/TrainingRunDetail';
import { Analysis } from './pages/Analysis';
import { History } from './pages/History';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Play />} />
          <Route path="/play" element={<Play />} />
          <Route path="/train" element={<TrainEval />} />
          <Route path="/training" element={<TrainingHistory />} />
          <Route path="/training/:runId" element={<TrainingRunDetail />} />
          <Route path="/analysis/:gameId" element={<Analysis />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
