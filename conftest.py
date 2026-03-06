"""
Root conftest.py — sets up sys.path so that:
  - worker_bridge (in browser_python/) is importable
  - engine.advanced_metrics (in top-level engine/) takes precedence over
    browser_python/engine/ which does not have that module

The project root must appear before browser_python on sys.path so that
`import engine` resolves to the top-level engine/ package (which contains
advanced_metrics, etc.), while worker_bridge itself is still importable from
browser_python/.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BROWSER_PYTHON = os.path.join(PROJECT_ROOT, "browser_python")

# Project root first → top-level engine/ found (has advanced_metrics.py)
# browser_python second → worker_bridge found
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BROWSER_PYTHON not in sys.path:
    sys.path.append(BROWSER_PYTHON)
