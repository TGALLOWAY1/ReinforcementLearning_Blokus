import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Home } from './pages/Home';
import { Play } from './pages/Play';
import { TrainEval } from './pages/TrainEval';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/play" element={<Play />} />
          <Route path="/train" element={<TrainEval />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
