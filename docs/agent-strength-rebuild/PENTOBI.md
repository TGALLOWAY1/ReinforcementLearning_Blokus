# Pentobi as external anchor and rules cross-check

Pentobi (Markus Enzenberger, GPL-3, https://github.com/enz/pentobi) is a mature
MCTS Blokus program. Decision D-020 permits using it; it serves three purposes
here: a fixed external yardstick at several strength levels (protocol v3
anchors), an independent rules cross-check (`all_legal`), and a stand-in for a
competent human until real human games are logged.

## Build (GTP engine only, no Qt) — verified 2026-09-07 on macOS, Apple clang 17

```bash
git clone --depth 1 --branch v30.3 https://github.com/enz/pentobi ~/.local/src/pentobi
cmake -S ~/.local/src/pentobi -B ~/.local/src/pentobi/build \
      -DCMAKE_BUILD_TYPE=Release -DPENTOBI_BUILD_GUI=OFF -DPENTOBI_BUILD_GTP=ON
cmake --build ~/.local/src/pentobi/build -j8 --target pentobi-gtp
mkdir -p ~/.local/opt/pentobi/bin ~/.local/bin
cp ~/.local/src/pentobi/build/pentobi_gtp/pentobi-gtp ~/.local/opt/pentobi/bin/
ln -sf ~/.local/opt/pentobi/bin/pentobi-gtp ~/.local/bin/pentobi-gtp
pentobi-gtp --version   # Pentobi 30.3
```

Build takes seconds (tag v30.3, commit 58356d06). The adapter finds the binary
via `$PENTOBI_GTP`, `PATH`, `~/.local/bin/pentobi-gtp` or
`~/.local/opt/pentobi/bin/pentobi-gtp` (`agents/pentobi_agent.py::find_binary`).
Opening books are disabled (`--nobook`) so play depends only on the seed.

## Protocol facts (verified against the build)

| Item | Value |
|---|---|
| Classic 4-colour 20×20 | `set_game Blokus` (also CLI `--game classic`, the default) |
| Colours, play order | `1` Blue, `2` Yellow, `3` Red, `4` Green; corners a20, t20, t1, a1 |
| Mapping to this engine | RED→1, BLUE→2, YELLOW→3, GREEN→4 (same corners, same order) |
| Coordinates | `<col a..t><row 1..20>`, row 20 at the top: repo `(row, col)` → `chr(97+col) + str(20-row)` |
| Move | comma-separated points, e.g. `play 1 a20,b20,b19`; `genmove 1` returns the same form or `pass` |
| All legal moves | `all_legal <colour>` (one move per line) |
| Illegal play | `? illegal move` / `? piece already played` / `? invalid move ` |
| Level | CLI `--level 1..9` (max simulations ≈ 3, 30, 90, 181, 667, 5028, 69809, 349044, 1745221, scaled by move number) |
| Seed | CLI `--seed N` (fresh process per game is reproducible with `--threads 1`) |
| Score | `final_score` → four scores in colour order |

Per-move latency (single thread, no book, this Mac): levels 1–5 ≈ 15–35 ms;
level 7 ≈ 0.6 s (opening) to 1.9 s (midgame); level 9 ≈ 11–40 s.

## Use in the repo

- Arena agent type `pentobi` with `params: {level: N, use_book: false}`
  (`analytics/tournament/arena_runner.build_agent`); one process per game.
- Rules cross-check: `tests/test_pentobi_agent.py::test_engine_legal_moves_match_pentobi_all_legal`
  compares the engine's legal-move sets with `all_legal` on every ply of seeded
  games (skipped when the binary is absent; `BLOKUS_PENTOBI_GAMES` scales it).
- Strength of each level relative to the repo's agents is measured, not
  assumed: see EXPERIMENT_LOG.md (EXP-016) once run.
