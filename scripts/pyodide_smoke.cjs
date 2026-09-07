// Headless smoke test of the Pyodide bundle: loads frontend/public/blokus_core.zip
// under node + pyodide, checks the piece catalogue, plays four seeded random games
// on the bundled engine (every colour must place the Z-pentomino at least once),
// and drives worker_bridge for three turns. Run: node scripts/pyodide_smoke.cjs
// (requires frontend/node_modules). Prints a JSON report.
const fs = require("fs");
const { loadPyodide } = require(require("path").join(__dirname, "..", "frontend", "node_modules", "pyodide"));
const ZIP = require("path").join(__dirname, "..", "frontend", "public", "blokus_core.zip");
const t0 = Date.now();
(async () => {
  const py = await loadPyodide();
  await py.loadPackage("numpy");
  py.unpackArchive(fs.readFileSync(ZIP).buffer, "zip");
  const out = await py.runPythonAsync(`
import sys, json
if "." not in sys.path: sys.path.append(".")
import numpy as np
from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator
from engine.pieces import PieceGenerator, ALL_PIECE_ORIENTATIONS, normalize_offsets, shape_to_offsets
from agents.random_agent import RandomAgent
import worker_bridge
from worker_bridge import bridge

gen = LegalMoveGenerator()
pieces = PieceGenerator.get_all_pieces()
p10 = next(p for p in pieces if p.id == 10)
report = {"piece10_name": p10.name, "piece10_cells": int(p10.shape.sum()),
          "total_squares": int(sum(p.shape.sum() for p in pieces)),
          "orientations_total": sum(len(v) for v in ALL_PIECE_ORIENTATIONS.values())}

# bridge orientation mapping for piece 10: every frontend orientation must hit a real backend orientation
backend = {tuple(o.offsets) for o in ALL_PIECE_ORIENTATIONS[10]}
misses = []
for fidx in range(8):
    shape = p10.shape.copy()
    if fidx >= 4: shape = np.fliplr(shape)
    for _ in range(fidx % 4): shape = np.rot90(shape)
    if tuple(normalize_offsets(shape_to_offsets(shape))) not in backend: misses.append(fidx)
report["piece10_frontend_orientations_unmapped"] = misses
report["piece10_backend_orientations"] = len(backend)

# seeded random games on the bundled engine
z5_by_colour = {p.name: 0 for p in Player}
games = []
for seed in (1, 2, 3, 4):
    board = Board(); agent = RandomAgent(seed=seed)
    order = list(Player); idle = 0; ply = 0; placements = 0
    while idle < 4 and ply < 400:
        pl = order[ply % 4]
        moves = gen.get_legal_moves(board, pl)
        if moves:
            idle = 0
            mv = agent.select_action(board, pl, moves)
            pos = mv.get_positions(gen.piece_orientations_cache[mv.piece_id])
            if mv.piece_id == 10:
                assert len(pos) == 5, len(pos)
                z5_by_colour[pl.name] += 1
            assert board.place_piece(pos, pl, mv.piece_id)
            placements += 1
        else:
            idle += 1
        ply += 1
    games.append({"seed": seed, "placements": placements,
                  "scores": {p.name: int(board.get_score(p)) for p in Player}})
report["games"] = games
report["z5_placements_by_colour"] = z5_by_colour

# bridge: standard scoring default and a game init with the 'champion' spec at a tiny budget
spec = {"type":"mcts","thinkingTimeMs":100,"mcts":{"rollout_policy":"greedy_sample","rollout_cutoff_depth":12,"heuristic_move_ordering":True,"iterations":5}}
cfg = {"players":[{"player":p,"agent_type":"mcts","agent_config":{"profile":"champion","time_budget_ms":100,"champion":spec}} for p in ("RED","BLUE","GREEN","YELLOW")],"auto_start":True}
st = bridge.init_game(cfg)
report["bridge_scoring_mode"] = st.get("scoring_mode")
msgs = []
for i in range(3):
    r = bridge.advance_turn(False)
    msgs.append(r.get("message"))
report["bridge_turns"] = msgs
report["bridge_pieces_used_sizes"] = {p: [int(PieceGenerator.get_piece_by_id(i).size) for i in bridge.game.board.player_pieces_used[Player[p]]] for p in ("RED","BLUE","GREEN")}
json.dumps(report)
`);
  console.log(out);
  console.log(`[smoke] total ${((Date.now()-t0)/1000).toFixed(1)}s`);
})().catch(e => { console.error("[smoke] ERROR", e); process.exit(1); });
