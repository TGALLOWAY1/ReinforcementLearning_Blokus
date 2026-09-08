"""M4 wiring: the web backend serves Pentobi seats natively, speaks the
frontend's orientation indices when asked, and logs every game for the human
protocol."""

import asyncio
import json
import os
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from agents import pentobi_agent as pa
from engine.board import Player as EnginePlayer
from engine.orientation_map import engine_to_frontend, frontend_to_engine
from schemas.game_state import AgentType, GameConfig, Move, MoveRequest, Player, PlayerConfig
from webapi.app import GameManager, create_app
from webapi.game_log import load_game_log
from webapi.gameplay_agent_factory import PentobiGameplayAdapter
from webapi.profile import APP_PROFILE_DEPLOY, APP_PROFILE_RESEARCH

needs_binary = pytest.mark.skipif(pa.find_binary() is None, reason="pentobi-gtp not installed")


@contextmanager
def _env(name, value):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _pentobi_config(game_id, *, level=1, human="RED", **extra):
    players = []
    for colour in ("RED", "BLUE", "GREEN", "YELLOW"):
        if colour == human:
            players.append(PlayerConfig(player=Player(colour), agent_type=AgentType.HUMAN, agent_config={}))
        else:
            players.append(PlayerConfig(player=Player(colour), agent_type=AgentType.PENTOBI,
                                        agent_config={"level": level, "seed": 1, "time_budget_ms": 10_000}))
    return GameConfig(game_id=game_id, players=players, auto_start=True, scoring_mode="standard", **extra)


# --- agent construction ------------------------------------------------------
@needs_binary
def test_research_profile_builds_served_pentobi_adapters(tmp_path):
    with _env("GAME_LOG_DIR", str(tmp_path)):
        manager = GameManager(app_profile=APP_PROFILE_RESEARCH)
        manager.create_game(_pentobi_config("g-build", level=1))
    agents = manager.agent_instances["g-build"]
    assert agents[EnginePlayer.RED] is None
    for colour in (EnginePlayer.BLUE, EnginePlayer.GREEN, EnginePlayer.YELLOW):
        assert isinstance(agents[colour], PentobiGameplayAdapter)
        assert agents[colour].budget.level == 1 and agents[colour].budget.cap_ms == 10_000
    manager.close_game_agents("g-build")


def test_available_agents_lists_pentobi_only_when_the_binary_exists(monkeypatch):
    manager = GameManager(app_profile=APP_PROFILE_RESEARCH)
    monkeypatch.setattr(pa, "find_binary", lambda explicit=None: "/usr/bin/pentobi-gtp")
    entry = next(a for a in manager.get_available_agents() if a.type == AgentType.PENTOBI)
    assert entry.config_schema["level"] == {"min": 1, "max": 9, "default": 3}
    assert entry.config_schema["time_budget_ms"]["max"] == 10_000
    assert entry.config_schema["available"] is True
    assert "binary" not in entry.config_schema   # never expose the server's filesystem layout
    monkeypatch.setattr(pa, "find_binary", lambda explicit=None: None)
    assert all(a.type != AgentType.PENTOBI for a in manager.get_available_agents())


def test_deploy_agent_listing_includes_pentobi_when_available(monkeypatch):
    monkeypatch.setattr(pa, "find_binary", lambda explicit=None: "/usr/bin/pentobi-gtp")
    app = create_app(profile=APP_PROFILE_DEPLOY, include_research_routes=False)
    types = {a["type"] for a in TestClient(app).get("/api/agents").json()}
    assert types == {"mcts", "human", "pentobi"}


# --- orientation space ---------------------------------------------------------
def test_frontend_orientation_space_translates_state_and_moves(tmp_path):
    with _env("GAME_LOG_DIR", ""):
        manager = GameManager(app_profile=APP_PROFILE_RESEARCH)
        config = GameConfig(
            game_id="g-ori",
            players=[PlayerConfig(player=Player.RED, agent_type=AgentType.HUMAN, agent_config={}),
                     PlayerConfig(player=Player.BLUE, agent_type=AgentType.HUMAN, agent_config={})],
            auto_start=True, scoring_mode="standard", orientation_space="frontend")
        manager.create_game(config)
        game = manager.games["g-ori"]["game"]
        engine_moves = game.get_legal_moves(EnginePlayer.RED)
        target = next(m for m in engine_moves if engine_to_frontend(m.piece_id, m.orientation) != m.orientation)
        state = manager._get_game_state("g-ori")
        assert state.orientation_space == "frontend"
        listed = {(m.piece_id, m.orientation, m.anchor_row, m.anchor_col) for m in state.legal_moves}
        assert (target.piece_id, engine_to_frontend(target.piece_id, target.orientation),
                target.anchor_row, target.anchor_col) in listed
        assert (target.piece_id, target.orientation, target.anchor_row, target.anchor_col) not in listed or \
            frontend_to_engine(target.piece_id, target.orientation) == target.orientation

        frontend_ori = engine_to_frontend(target.piece_id, target.orientation)
        resp = asyncio.run(manager.make_move("g-ori", MoveRequest(
            move=Move(piece_id=target.piece_id, orientation=frontend_ori,
                      anchor_row=target.anchor_row, anchor_col=target.anchor_col),
            player=Player.RED)))
        assert resp.success, resp.message
        expected = target.get_positions(game.move_generator.piece_orientations_cache[target.piece_id])
        for pos in expected:
            assert game.board.grid[pos.row][pos.col] == EnginePlayer.RED.value
        assert resp.game_state.last_move["orientation"] == frontend_ori
        assert resp.game_state.last_move["engine_orientation"] == target.orientation


def test_engine_orientation_space_is_the_default_and_untranslated():
    with _env("GAME_LOG_DIR", ""):
        manager = GameManager(app_profile=APP_PROFILE_RESEARCH)
        manager.create_game(GameConfig(
            game_id="g-eng",
            players=[PlayerConfig(player=Player.RED, agent_type=AgentType.HUMAN, agent_config={}),
                     PlayerConfig(player=Player.BLUE, agent_type=AgentType.HUMAN, agent_config={})],
            auto_start=True))
        game = manager.games["g-eng"]["game"]
        engine = {(m.piece_id, m.orientation, m.anchor_row, m.anchor_col) for m in game.get_legal_moves(EnginePlayer.RED)}
        state = manager._get_game_state("g-eng")
        assert state.orientation_space == "engine"
        assert {(m.piece_id, m.orientation, m.anchor_row, m.anchor_col) for m in state.legal_moves} == engine


# --- game log --------------------------------------------------------------------
@needs_binary
def test_game_log_records_seats_moves_and_result(tmp_path):
    with _env("GAME_LOG_DIR", str(tmp_path)):
        manager = GameManager(app_profile=APP_PROFILE_RESEARCH)
        manager.create_game(_pentobi_config("g-log", level=1, human="BLUE"))
        path = tmp_path / "g-log.json"
        rec = load_game_log(path)
        assert rec["seats"] == {"RED": "pentobi", "BLUE": "human", "GREEN": "pentobi", "YELLOW": "pentobi"}
        assert rec["human_seats"] == ["BLUE"] and rec["scoring_mode"] == "standard"
        assert rec["meta"]["app_profile"] == "research"
        assert rec["players"][0]["agent_config"]["level"] == 1

        # RED (pentobi) moves first
        resp = asyncio.run(manager.advance_turn("g-log"))
        assert resp.success and resp.message == "Agent moved"
        rec = load_game_log(path)
        assert rec["moves"][0]["player"] == "RED" and rec["moves"][0]["is_human"] is False
        assert rec["moves"][0]["stats"]["engine"] == "pentobi" and rec["moves"][0]["stats"]["level"] == 1
        assert rec["moves"][0]["stats"]["fallback"] is None
        assert "timeSpentMs" in rec["moves"][0]["stats"]

        # BLUE (human) moves
        game = manager.games["g-log"]["game"]
        human_move = game.get_legal_moves(EnginePlayer.BLUE)[0]
        resp = asyncio.run(manager.make_move("g-log", MoveRequest(
            move=Move(piece_id=human_move.piece_id, orientation=human_move.orientation,
                      anchor_row=human_move.anchor_row, anchor_col=human_move.anchor_col), player=Player.BLUE)))
        assert resp.success
        rec = load_game_log(path)
        assert rec["moves"][1]["player"] == "BLUE" and rec["moves"][1]["is_human"] is True
        assert rec["moves"][1]["stats"]["timeSpentMs"] >= 0

        asyncio.run(manager.force_finish_game("g-log"))
        rec = load_game_log(path)
        assert rec["status"] == "finished" and set(rec["result"]["scores"]) == {"RED", "BLUE", "GREEN", "YELLOW"}
        assert rec["result"]["ranks"]["RED"] in (1, 2, 3, 4)
        manager.close_game_agents("g-log")


def test_human_move_that_ends_the_game_finishes_the_log(tmp_path, monkeypatch):
    with _env("GAME_LOG_DIR", str(tmp_path)):
        manager = GameManager(app_profile=APP_PROFILE_RESEARCH)
        manager.create_game(GameConfig(
            game_id="g-end",
            players=[PlayerConfig(player=Player.RED, agent_type=AgentType.HUMAN, agent_config={}),
                     PlayerConfig(player=Player.BLUE, agent_type=AgentType.HUMAN, agent_config={})],
            auto_start=True, scoring_mode="standard"))
        game = manager.games["g-end"]["game"]
        move = game.get_legal_moves(EnginePlayer.RED)[0]
        monkeypatch.setattr(game, "is_game_over", lambda: True)
        resp = asyncio.run(manager.make_move("g-end", MoveRequest(
            move=Move(piece_id=move.piece_id, orientation=move.orientation,
                      anchor_row=move.anchor_row, anchor_col=move.anchor_col), player=Player.RED)))
        assert resp.success and resp.game_state.status == "finished"
        assert load_game_log(tmp_path / "g-end.json")["status"] == "finished"


def test_game_log_is_off_when_the_directory_is_blank(tmp_path):
    with _env("GAME_LOG_DIR", ""):
        manager = GameManager(app_profile=APP_PROFILE_RESEARCH)
        manager.create_game(GameConfig(
            game_id="g-off",
            players=[PlayerConfig(player=Player.RED, agent_type=AgentType.HUMAN, agent_config={}),
                     PlayerConfig(player=Player.BLUE, agent_type=AgentType.HUMAN, agent_config={})],
            auto_start=True))
        assert manager.game_log is None


# --- HTTP round trip ---------------------------------------------------------------
@needs_binary
def test_http_round_trip_creates_a_pentobi_game_and_advances_an_ai_turn(tmp_path):
    with _env("GAME_LOG_DIR", str(tmp_path)):
        app = create_app(profile=APP_PROFILE_RESEARCH, include_research_routes=False)
        client = TestClient(app)
        payload = json.loads(_pentobi_config("g-http", level=1, human="BLUE",
                                             orientation_space="frontend").json())
        resp = client.post("/api/games", json=payload)
        assert resp.status_code == 200, resp.text
        state = resp.json()["game_state"]
        assert [p["agent_type"] for p in state["players"]] == ["pentobi", "human", "pentobi", "pentobi"]
        assert state["orientation_space"] == "frontend"
        resp = client.post("/api/games/g-http/advance_turn")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] and body["message"] == "Agent moved"
        assert body["game_state"]["current_player"] == "BLUE"
        assert body["game_state"]["mcts_stats"]["engine"] == "pentobi"
        assert body["game_state"]["last_move"]["player"] == "RED"
        assert (tmp_path / "g-http.json").exists()
        resp = client.post("/api/games/g-http/finish")
        assert resp.status_code == 200


# --- end-of-game persistence must await the async database accessor ------------
def test_end_game_awaits_the_database_accessor(monkeypatch, tmp_path):
    import webapi.app as app_module

    calls = []

    class _Coll:
        async def update_one(self, *a, **k):
            calls.append(("update_one", a[0]))

        async def delete_many(self, *a, **k):
            calls.append(("delete_many", a[0]))

        async def insert_many(self, docs):
            calls.append(("insert_many", len(docs)))

    class _Db:
        game_records = _Coll()
        move_records = _Coll()

    async def fake_get_database():
        return _Db()

    monkeypatch.setattr(app_module, "get_database", fake_get_database)
    monkeypatch.setattr(app_module, "_mongo_ready", True)
    with _env("GAME_LOG_DIR", ""):
        manager = GameManager(app_profile=APP_PROFILE_RESEARCH)
        manager.create_game(GameConfig(
            game_id="g-db",
            players=[PlayerConfig(player=Player.RED, agent_type=AgentType.HUMAN, agent_config={}),
                     PlayerConfig(player=Player.BLUE, agent_type=AgentType.HUMAN, agent_config={})],
            auto_start=True))
        asyncio.run(manager.force_finish_game("g-db"))
    assert ("update_one", {"game_id": "g-db"}) in calls


# --- review fixes -----------------------------------------------------------------
@needs_binary
def test_research_profile_normalises_pentobi_seats_like_deploy(tmp_path):
    """The human protocol runs in the research profile: the 10 s cap, seed
    coercion and the binary-stripping must not depend on the deploy normaliser."""
    with _env("GAME_LOG_DIR", ""):
        manager = GameManager(app_profile=APP_PROFILE_RESEARCH)
        manager.create_game(GameConfig(game_id="g-norm", auto_start=True, scoring_mode="standard", players=[
            PlayerConfig(player=Player.RED, agent_type=AgentType.PENTOBI,
                         agent_config={"level": 1, "seed": "7", "binary": "/nope/pentobi", "use_book": True}),
            PlayerConfig(player=Player.BLUE, agent_type=AgentType.HUMAN, agent_config={})]))
        cfg = manager.games["g-norm"]["config"].players[0].agent_config
        assert cfg == {"level": 1, "seed": 7, "time_budget_ms": 10_000}
        adapter = manager.agent_instances["g-norm"][EnginePlayer.RED]
        assert adapter.agent.seed == 7 and adapter.agent.binary is None
        resp = asyncio.run(manager.advance_turn("g-norm"))
        assert resp.success and resp.game_state.mcts_stats["timeBudgetMs"] == 10_000
        manager.close_game_agents("g-norm")


def test_research_profile_rejects_invalid_pentobi_config():
    from fastapi import HTTPException

    with _env("GAME_LOG_DIR", ""):
        manager = GameManager(app_profile=APP_PROFILE_RESEARCH)
        with pytest.raises(HTTPException) as exc:
            manager.create_game(GameConfig(game_id="g-bad", auto_start=True, players=[
                PlayerConfig(player=Player.RED, agent_type=AgentType.PENTOBI, agent_config={"level": 3, "seed": "abc"}),
                PlayerConfig(player=Player.BLUE, agent_type=AgentType.HUMAN, agent_config={})]))
        assert exc.value.status_code == 400 and "seed" in str(exc.value.detail)
        assert "g-bad" not in manager.games


@needs_binary
def test_moves_for_agent_seats_and_out_of_turn_are_rejected(tmp_path):
    with _env("GAME_LOG_DIR", str(tmp_path)):
        manager = GameManager(app_profile=APP_PROFILE_RESEARCH)
        manager.create_game(_pentobi_config("g-seat", level=1, human="BLUE"))
        game = manager.games["g-seat"]["game"]
        move = game.get_legal_moves(EnginePlayer.RED)[0]
        req = lambda player: MoveRequest(move=Move(piece_id=move.piece_id, orientation=move.orientation,
                                                   anchor_row=move.anchor_row, anchor_col=move.anchor_col), player=player)
        resp = asyncio.run(manager.make_move("g-seat", req(Player.RED)))      # RED is a Pentobi seat
        assert resp.success is False and "agent" in resp.message.lower()
        resp = asyncio.run(manager.make_move("g-seat", req(Player.BLUE)))     # human, but not BLUE's turn
        assert resp.success is False and "turn" in resp.message.lower()
        assert game.get_move_count() == 0
        assert load_game_log(tmp_path / "g-seat.json")["moves"] == []
        manager.close_game_agents("g-seat")


def test_replay_uses_engine_orientation_for_frontend_space_games():
    import webapi.app as app_module

    with _env("GAME_LOG_DIR", ""):
        manager = GameManager(app_profile=APP_PROFILE_RESEARCH)
        manager.create_game(GameConfig(game_id="g-replay", auto_start=True, scoring_mode="standard",
                                       orientation_space="frontend", players=[
            PlayerConfig(player=Player.RED, agent_type=AgentType.HUMAN, agent_config={}),
            PlayerConfig(player=Player.BLUE, agent_type=AgentType.HUMAN, agent_config={})]))
        game = manager.games["g-replay"]["game"]
        target = next(m for m in game.get_legal_moves(EnginePlayer.RED)
                      if engine_to_frontend(m.piece_id, m.orientation) != m.orientation)
        resp = asyncio.run(manager.make_move("g-replay", MoveRequest(
            move=Move(piece_id=target.piece_id, orientation=engine_to_frontend(target.piece_id, target.orientation),
                      anchor_row=target.anchor_row, anchor_col=target.anchor_col), player=Player.RED)))
        assert resp.success
        app_module.game_manager.games["g-replay"] = manager.games["g-replay"]
        try:
            replayed, _doc, _moves = asyncio.run(app_module._replay_to_index("g-replay", 1))
        finally:
            app_module.game_manager.games.pop("g-replay", None)
        assert (replayed.board.grid == game.board.grid).all()
        assert int((replayed.board.grid != 0).sum()) == len(
            target.get_positions(game.move_generator.piece_orientations_cache[target.piece_id]))


def test_idle_games_are_evicted_and_their_agents_closed():
    closed = []

    class _Agent:
        def close(self):
            closed.append(True)

    with _env("GAME_LOG_DIR", ""):
        manager = GameManager(app_profile=APP_PROFILE_RESEARCH)
        manager.create_game(GameConfig(game_id="g-idle", auto_start=True, players=[
            PlayerConfig(player=Player.RED, agent_type=AgentType.HUMAN, agent_config={}),
            PlayerConfig(player=Player.BLUE, agent_type=AgentType.HUMAN, agent_config={})]))
        manager.agent_instances["g-idle"][EnginePlayer.BLUE] = _Agent()
        assert manager.evict_idle_games(max_idle_seconds=3600) == []
        manager.games["g-idle"]["updated_at"] = manager.games["g-idle"]["updated_at"].replace(year=2000)
        assert manager.evict_idle_games(max_idle_seconds=3600) == ["g-idle"]
        assert closed == [True] and "g-idle" not in manager.games and "g-idle" not in manager.agent_instances


def test_delete_route_closes_agents_and_forgets_the_game():
    closed = []

    class _Agent:
        def close(self):
            closed.append(True)

    with _env("GAME_LOG_DIR", ""):
        app = create_app(profile=APP_PROFILE_RESEARCH, include_research_routes=False)
        client = TestClient(app)
        resp = client.post("/api/games", json={"game_id": "g-del", "auto_start": True, "players": [
            {"player": "RED", "agent_type": "human", "agent_config": {}},
            {"player": "BLUE", "agent_type": "human", "agent_config": {}}]})
        assert resp.status_code == 200
        import webapi.app as app_module
        app_module.game_manager.agent_instances["g-del"][EnginePlayer.BLUE] = _Agent()
        resp = client.delete("/api/games/g-del")
        assert resp.status_code == 200 and closed == [True]
        assert client.get("/api/games/g-del").status_code == 404
        assert client.delete("/api/games/g-del").status_code == 404


def test_create_app_closes_agents_of_games_it_discards():
    import webapi.app as app_module

    closed = []

    class _Agent:
        def close(self):
            closed.append(True)

    app_module.game_manager.games["g-stale"] = {"status": "in_progress"}
    app_module.game_manager.agent_instances["g-stale"] = {EnginePlayer.RED: _Agent()}
    create_app(profile=APP_PROFILE_RESEARCH, include_research_routes=False)
    assert closed == [True] and "g-stale" not in app_module.game_manager.games


@needs_binary
def test_whole_game_over_http_with_a_human_seat_in_frontend_space(tmp_path):
    """The path the UI actually drives: /move and /pass for the human seat in
    frontend orientation indices, /advance_turn for the three Pentobi seats,
    the game ending through advance_turn when every seat is blocked."""
    from agents.greedy_agent import GreedyAgent

    with _env("GAME_LOG_DIR", str(tmp_path)):
        app = create_app(profile=APP_PROFILE_RESEARCH, include_research_routes=False)
        client = TestClient(app)
        payload = json.loads(_pentobi_config("g-full", level=1, human="GREEN", orientation_space="frontend").json())
        assert client.post("/api/games", json=payload).status_code == 200
        import webapi.app as app_module
        game = app_module.game_manager.games["g-full"]["game"]
        human = GreedyAgent(seed=0)
        human_moves = human_passes = 0
        for _ in range(400):
            state = client.get("/api/games/g-full").json()
            if state["status"] == "finished":
                break
            if state["current_player"] != "GREEN":
                resp = client.post("/api/games/g-full/advance_turn").json()
                assert resp["success"], resp["message"]
                continue
            legal = game.get_legal_moves(EnginePlayer.GREEN)
            if not legal:
                assert client.post("/api/games/g-full/pass", json={"player": "GREEN"}).json()["success"]
                human_passes += 1
                continue
            move = human.select_action(game.board, EnginePlayer.GREEN, legal)
            resp = client.post("/api/games/g-full/move", json={"player": "GREEN", "move": {
                "piece_id": move.piece_id, "orientation": engine_to_frontend(move.piece_id, move.orientation),
                "anchor_row": move.anchor_row, "anchor_col": move.anchor_col}}).json()
            assert resp["success"], resp["message"]
            assert resp["game_state"]["last_move"]["engine_orientation"] == move.orientation
            human_moves += 1
        state = client.get("/api/games/g-full").json()
        assert state["status"] == "finished" and human_moves >= 5 and human_passes >= 1
        rec = load_game_log(tmp_path / "g-full.json")
        assert rec["status"] == "finished" and rec["result"]["ranks"]["GREEN"] in (1, 2, 3, 4)
        assert any(m["is_pass"] and m["is_human"] for m in rec["moves"])
        assert sum(1 for m in rec["moves"] if m["is_human"] and not m["is_pass"]) == human_moves
        assert rec["fallbacks"] == []
        for adapter in app_module.game_manager.agent_instances["g-full"].values():
            if adapter is not None:
                assert adapter.agent._engine is None   # closed at game end
