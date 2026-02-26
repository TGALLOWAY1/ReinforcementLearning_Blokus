import { loadPyodide } from 'pyodide';

// We need to define the type for this worker if TS doesn't know it's a dedicated worker
const ctx: Worker = self as any;

let pyodide: any = null;
let bridge: any = null;

async function initPyodide() {
    console.log("[Worker] Initializing Pyodide...");

    // We fetch pyodide from unpkg or load it from node_modules. 
    // Since we installed pyodide, Vite will serve it if we import `loadPyodide`.
    pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/"
    });
    await pyodide.loadPackage("numpy");

    console.log("[Worker] Pyodide initialized. Fetching blokus_core.zip...");
    // Fetch the python source
    const response = await fetch('/blokus_core.zip');
    if (!response.ok) {
        throw new Error(`Failed to load blokus_core.zip: ${response.status}`);
    }
    const buffer = await response.arrayBuffer();

    console.log("[Worker] Unpacking python core...");
    // Unpack ZIP into Pyodide virtual filesystem
    pyodide.unpackArchive(buffer, "zip");

    console.log("[Worker] Importing worker_bridge.py...");
    // Import our bridge logic
    await pyodide.runPythonAsync(`
        import sys
        # ensure current dir is in path
        if "." not in sys.path:
            sys.path.append(".")
        import worker_bridge
    `);

    // Get a reference to the global `bridge` object
    bridge = pyodide.globals.get("worker_bridge").bridge;

    console.log("[Worker] Initialization Complete.");

    // Notify main thread we are ready
    ctx.postMessage({ type: 'ready' });
}

ctx.addEventListener('message', async (e) => {
    const data = e.data;
    if (!pyodide || !bridge) {
        console.error("[Worker] Pyodide not ready yet.");
        ctx.postMessage({ type: 'error', error: "Pyodide not ready" });
        return;
    }

    try {
        switch (data.type) {
            case 'init_game': {
                console.log("[Worker] init_game", data.config);
                const configDict = pyodide.toPy(data.config);
                const pyState = bridge.init_game(configDict);
                const state = pyState.toJs({ dict_converter: Object.fromEntries });

                // Cleanup Pyodide proxies manually to prevent memory leaks
                pyState.destroy();
                configDict.destroy();

                ctx.postMessage({ type: 'state_update', state });
                break;
            }
            case 'make_move': {
                const { piece_id, orientation, anchor_row, anchor_col } = data.move;
                const pyResp = bridge.make_move(piece_id, orientation, anchor_row, anchor_col);
                const response = pyResp.toJs({ dict_converter: Object.fromEntries });
                pyResp.destroy();
                ctx.postMessage({ type: 'move_response', response });
                break;
            }
            case 'pass_turn': {
                const pyResp = bridge.pass_turn();
                const response = pyResp.toJs({ dict_converter: Object.fromEntries });
                pyResp.destroy();
                ctx.postMessage({ type: 'move_response', response });
                break;
            }
            case 'advance_turn': {
                const pyResp = bridge.advance_turn();
                const response = pyResp.toJs({ dict_converter: Object.fromEntries });
                pyResp.destroy();
                ctx.postMessage({ type: 'move_response', response });
                break;
            }
            case 'load_game': {
                const pyResp = bridge.load_game(pyodide.toPy(data.history));
                const state = pyResp.toJs({ dict_converter: Object.fromEntries });
                pyResp.destroy();
                ctx.postMessage({ type: 'state_update', state });
                break;
            }
        }
    } catch (err: any) {
        console.error("[Worker] Python Execution Error: ", err);
        ctx.postMessage({ type: 'error', error: err.toString() });
    }
});

// Start initialization on load
initPyodide().catch(err => {
    console.error("[Worker] Failed to init pyodide", err);
    ctx.postMessage({ type: 'init_error', error: err.toString() });
});
