#!/usr/bin/env python3
"""
Run the Blokus web API server.
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run the app
import uvicorn

if __name__ == "__main__":
    print("Starting Blokus Web API server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at:  http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "webapi.app:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True  # Enable auto-reload for development
    )

# cd webapi
# python run_server.py