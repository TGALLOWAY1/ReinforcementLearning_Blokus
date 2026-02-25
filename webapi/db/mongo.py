"""
MongoDB connection module for Blokus RL Web API.

This module provides a centralized MongoDB connection using Motor (async MongoDB driver)
for use with FastAPI. The connection is established once at startup and reused across requests.

Environment Variables:
    MONGODB_URI: MongoDB connection string (default: mongodb://localhost:27017)
    MONGODB_DB_NAME: Database name (default: blokusdb)

Example local connection string:
    mongodb://localhost:27017/blokusdb

For production, use a connection string like:
    mongodb://username:password@host:port/database?authSource=admin
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Load .env file if present (development)
# Production should use OS environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Environment variables with defaults
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "blokusdb")

# Global client and database instances (singleton pattern)
# These are initialized in connect_to_mongo() and should be accessed via get_client() and get_database()
_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None

# Module-level client and db for direct access (initialized lazily)
# These will be set when connect_to_mongo() is called
client: Optional[AsyncIOMotorClient] = None
db: Optional[AsyncIOMotorDatabase] = None


async def connect_to_mongo() -> None:
    """
    Establish MongoDB connection at application startup.
    
    This should be called during FastAPI lifespan startup.
    The connection is stored as a global singleton and reused across requests.
    
    Raises:
        ConnectionFailure: If connection to MongoDB fails
        ServerSelectionTimeoutError: If MongoDB server is not reachable
    """
    global _client, _database, client, db
    
    if _client is not None:
        logger.warning("MongoDB client already initialized, skipping connection")
        return
    
    try:
        logger.info(f"Connecting to MongoDB: {MONGODB_URI.split('@')[-1] if '@' in MONGODB_URI else MONGODB_URI}")
        logger.info(f"Using database: {MONGODB_DB_NAME}")
        
        _client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        
        # Test connection with server_info()
        try:
            await _client.server_info()
            logger.info("✅ MongoDB connection successful")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise
        
        _database = _client[MONGODB_DB_NAME]
        
        # Set module-level client and db for direct access
        client = _client
        db = _database
        
        logger.info(f"Successfully connected to MongoDB database: {MONGODB_DB_NAME}")
        
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        logger.error("Please ensure MongoDB is running and MONGODB_URI is correct")
        _client = None
        _database = None
        client = None
        db = None
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error connecting to MongoDB: {e}")
        _client = None
        _database = None
        client = None
        db = None
        raise


async def close_mongo_connection() -> None:
    """
    Close MongoDB connection at application shutdown.
    
    This should be called during FastAPI lifespan shutdown.
    """
    global _client, _database, client, db
    
    if _client is not None:
        logger.info("Closing MongoDB connection...")
        _client.close()
        _client = None
        _database = None
        client = None
        db = None
        logger.info("MongoDB connection closed")
    else:
        logger.debug("MongoDB client not initialized, skipping close")


async def get_database() -> AsyncIOMotorDatabase:
    """
    Get the MongoDB database instance. Tests connection and reconnects if needed.
    
    This should be called after connect_to_mongo() has been executed.
    
    Returns:
        AsyncIOMotorDatabase instance
        
    Raises:
        RuntimeError: If database connection fails and cannot be re-established
    """
    global _client, _database, client, db
    
    if _client is None or _database is None:
        try:
            await connect_to_mongo()
        except Exception as e:
            raise RuntimeError(f"MongoDB database not initialized and auto-connect failed: {e}")
            
    try:
        # Ping to check if connection is still alive (essential for Vercel frozen containers)
        await _client.admin.command('ping')
    except Exception as e:
        logger.warning(f"MongoDB ping failed, attempting to reconnect: {e}")
        try:
            # Clean up old client
            _client.close()
        except:
            pass
            
        _client = None
        _database = None
        client = None
        db = None
        
        try:
            await connect_to_mongo()
        except Exception as reconnect_e:
            raise RuntimeError(f"MongoDB database re-initialization failed: {reconnect_e}")
            
    if _database is None:
        raise RuntimeError("MongoDB database not initialized after reconnection attempt.")
        
    return _database


def get_client() -> Optional[AsyncIOMotorClient]:
    """
    Get the MongoDB client instance (for advanced use cases).
    
    Returns:
        AsyncIOMotorClient instance or None if not connected
    """
    return _client

