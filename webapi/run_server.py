"""
Script to run the Blokus RL Web API server
"""

import uvicorn
from app import app

if __name__ == "__main__":
    print("Starting Blokus RL Web API server...")
    print("API Documentation: http://localhost:8000/docs")
    print("WebSocket endpoint: ws://localhost:8000/ws/games/{game_id}")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
