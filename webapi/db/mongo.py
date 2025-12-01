"""
MongoDB connection module for Blokus RL Web API.

This module provides a centralized MongoDB connection using Motor (async MongoDB driver)
for use with FastAPI. The connection is established once at startup and reused across requests.

Environment Variables:
    MONGODB_URI: MongoDB connection string (default: mongodb://localhost:27017)
    MONGODB_DB_NAME: Database name (default: blokus_rl)

Example local connection string:
    mongodb://localhost:27017/blokus_rl

For production, use a connection string like:
    mongodb://username:password@host:port/database?authSource=admin
"""

import os
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

logger = logging.getLogger(__name__)

# Global client instance (singleton)
_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None


def get_mongodb_uri() -> str:
    """
    Get MongoDB URI from environment variable with fallback to default.
    
    Returns:
        MongoDB connection string
    """
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    logger.info(f"Using MongoDB URI: {uri.split('@')[-1] if '@' in uri else uri}")  # Log without credentials
    return uri


def get_mongodb_db_name() -> str:
    """
    Get MongoDB database name from environment variable with fallback to default.
    
    Returns:
        Database name
    """
    db_name = os.getenv("MONGODB_DB_NAME", "blokus_rl")
    logger.info(f"Using MongoDB database: {db_name}")
    return db_name


async def connect_to_mongo() -> None:
    """
    Establish MongoDB connection at application startup.
    
    This should be called during FastAPI lifespan startup.
    The connection is stored as a global singleton and reused across requests.
    
    Raises:
        ConnectionFailure: If connection to MongoDB fails
        ServerSelectionTimeoutError: If MongoDB server is not reachable
    """
    global _client, _database
    
    if _client is not None:
        logger.warning("MongoDB client already initialized, skipping connection")
        return
    
    try:
        uri = get_mongodb_uri()
        db_name = get_mongodb_db_name()
        
        logger.info("Connecting to MongoDB...")
        _client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
        
        # Test connection with a ping
        await _client.admin.command("ping")
        
        _database = _client[db_name]
        logger.info(f"Successfully connected to MongoDB database: {db_name}")
        
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        logger.error("Please ensure MongoDB is running and MONGODB_URI is correct")
        _client = None
        _database = None
        raise
    except Exception as e:
        logger.error(f"Unexpected error connecting to MongoDB: {e}")
        _client = None
        _database = None
        raise


async def close_mongo_connection() -> None:
    """
    Close MongoDB connection at application shutdown.
    
    This should be called during FastAPI lifespan shutdown.
    """
    global _client, _database
    
    if _client is not None:
        logger.info("Closing MongoDB connection...")
        _client.close()
        _client = None
        _database = None
        logger.info("MongoDB connection closed")
    else:
        logger.debug("MongoDB client not initialized, skipping close")


def get_database() -> AsyncIOMotorDatabase:
    """
    Get the MongoDB database instance.
    
    This should be called after connect_to_mongo() has been executed.
    
    Returns:
        AsyncIOMotorDatabase instance
        
    Raises:
        RuntimeError: If database connection has not been established
    """
    if _database is None:
        raise RuntimeError(
            "MongoDB database not initialized. "
            "Call connect_to_mongo() during application startup."
        )
    return _database


def get_client() -> Optional[AsyncIOMotorClient]:
    """
    Get the MongoDB client instance (for advanced use cases).
    
    Returns:
        AsyncIOMotorClient instance or None if not connected
    """
    return _client

