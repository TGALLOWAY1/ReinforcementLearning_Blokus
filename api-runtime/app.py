"""
Deploy-only API runtime entrypoint (gameplay routes only).
"""

from __future__ import annotations

import os

os.environ.setdefault("APP_PROFILE", "deploy")

from webapi.app import create_app  # noqa: E402

app = create_app(profile="deploy", include_research_routes=False)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
