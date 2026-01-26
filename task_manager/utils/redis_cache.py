"""
Redis Cache Manager - Task result caching with TTL support
"""

import json
import redis
from datetime import datetime
from typing import Optional, Dict, Any
from task_manager.utils.logger import get_logger

logger = get_logger(__name__)


class RedisCacheManager:
    """
    Redis-based cache manager for task results.
    
    Provides persistent caching of task inputs/outputs with automatic expiration.
    Uses Redis Hashes for structured storage and JSON serialization for complex data.
    
    Features:
    - Hash-based storage for efficient field access
    - Automatic TTL (24 hours default)
    - JSON serialization for complex data structures
    - Connection pooling and error handling
    - Graceful degradation when Redis unavailable
    
    Usage:
        cache = RedisCacheManager()
        
        # Cache a task result
        cache.cache_task_result(
            task_id="task_1.2",
            input_data={"query": "Karnataka districts"},
            output_data={"districts": ["Bangalore", "Mysore"]},
            agent_type="web_search_agent"
        )
        
        # Retrieve cached result
        cached = cache.get_cached_result("task_1.2")
        if cached:
            print(cached['output'])
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        decode_responses: bool = True,
        default_ttl: int = 86400  # 24 hours
    ):
        """
        Initialize Redis cache manager with connection pooling.
        
        Args:
            host: Redis server host (default: localhost)
            port: Redis server port (default: 6379)
            db: Redis database number (default: 0)
            password: Redis password if authentication required
            decode_responses: Auto-decode byte responses to strings
            default_ttl: Default TTL in seconds (default: 86400 = 24 hours)
        """
        self.default_ttl = default_ttl
        self.redis_available = False
        self.client: Optional[redis.Redis] = None
        
        try:
            # Create Redis client with connection pool
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=decode_responses,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.client.ping()
            self.redis_available = True
            logger.info(f"[REDIS] Connected to Redis at {host}:{port} (db={db})")
            
        except redis.ConnectionError as e:
            logger.warning(f"[REDIS] Connection failed: {str(e)}")
            logger.warning("[REDIS] Cache will operate in disabled mode (no caching)")
            self.redis_available = False
            
        except Exception as e:
            logger.error(f"[REDIS] Unexpected error during initialization: {str(e)}")
            logger.warning("[REDIS] Cache disabled")
            self.redis_available = False
    
    def cache_task_result(
        self,
        task_id: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        agent_type: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a task result in Redis using Hash storage.
        
        Stores the following fields in Redis Hash:
        - input: JSON-serialized input data
        - output: JSON-serialized output data
        - timestamp: ISO format timestamp of cache creation
        - agent_type: Type of agent that produced this result
        
        Args:
            task_id: Unique task identifier
            input_data: Task input parameters (will be JSON serialized)
            output_data: Task execution results (will be JSON serialized)
            agent_type: Name of the agent that executed this task
            ttl: Time-to-live in seconds (default: 86400)
            
        Returns:
            True if caching succeeded, False otherwise
            
        Example:
            success = cache.cache_task_result(
                task_id="task_1.2",
                input_data={"query": "Karnataka districts", "max_results": 10},
                output_data={"success": True, "results": [...]},
                agent_type="web_search_agent"
            )
        """
        if not self.redis_available or not self.client:
            logger.debug(f"[REDIS] Cache disabled - skipping cache for task {task_id}")
            return False
        
        try:
            # Build hash key
            cache_key = f"task:{task_id}"
            
            # Prepare hash fields with JSON serialization
            hash_data = {
                "input": json.dumps(input_data, default=str),
                "output": json.dumps(output_data, default=str),
                "timestamp": datetime.now().isoformat(),
                "agent_type": agent_type or "unknown"
            }
            
            # Store in Redis Hash
            self.client.hset(cache_key, mapping=hash_data)
            
            # Set TTL
            ttl_seconds = ttl if ttl is not None else self.default_ttl
            self.client.expire(cache_key, ttl_seconds)
            
            logger.info(f"[REDIS] ✓ Cached task result: {cache_key} (TTL: {ttl_seconds}s)")
            logger.debug(f"[REDIS] Cached fields: {list(hash_data.keys())}")
            
            return True
            
        except redis.RedisError as e:
            logger.error(f"[REDIS] Failed to cache task {task_id}: {str(e)}")
            return False
            
        except (TypeError, ValueError) as e:
            logger.error(f"[REDIS] JSON serialization error for task {task_id}: {str(e)}")
            return False
            
        except Exception as e:
            logger.error(f"[REDIS] Unexpected error caching task {task_id}: {str(e)}")
            return False
    
    def get_cached_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached task result from Redis.
        
        Fetches all hash fields and deserializes JSON strings back into
        Python dictionaries for input and output data.
        
        Args:
            task_id: Unique task identifier
            
        Returns:
            Dictionary containing:
            - input: Deserialized input data (dict)
            - output: Deserialized output data (dict)
            - timestamp: ISO format string of cache creation
            - agent_type: Type of agent that produced this result
            - ttl: Remaining time-to-live in seconds
            
            Returns None if:
            - Redis unavailable
            - Task not found in cache
            - Cache entry expired
            - Deserialization failed
            
        Example:
            cached = cache.get_cached_result("task_1.2")
            if cached:
                print(f"Agent: {cached['agent_type']}")
                print(f"Cached at: {cached['timestamp']}")
                print(f"Results: {cached['output']}")
                print(f"Expires in: {cached['ttl']} seconds")
        """
        if not self.redis_available or not self.client:
            logger.debug(f"[REDIS] Cache disabled - no cached result for task {task_id}")
            return None
        
        try:
            cache_key = f"task:{task_id}"
            
            # Check if key exists
            if not self.client.exists(cache_key):
                logger.debug(f"[REDIS] Cache miss: {cache_key}")
                return None
            
            # Retrieve all hash fields
            hash_data = self.client.hgetall(cache_key)  # type: ignore
            
            if not hash_data:
                logger.debug(f"[REDIS] Empty cache entry: {cache_key}")
                return None
            
            # Get remaining TTL
            ttl = self.client.ttl(cache_key)  # type: ignore
            
            # Deserialize JSON fields
            result = {
                "input": json.loads(hash_data.get("input", "{}")),  # type: ignore
                "output": json.loads(hash_data.get("output", "{}")),  # type: ignore
                "timestamp": hash_data.get("timestamp", ""),  # type: ignore
                "agent_type": hash_data.get("agent_type", "unknown"),  # type: ignore
                "ttl": ttl if ttl > 0 else 0  # type: ignore
            }
            
            logger.info(f"[REDIS] ✓ Cache hit: {cache_key} (TTL: {ttl}s)")
            logger.debug(f"[REDIS] Retrieved fields: {list(hash_data.keys())}")  # type: ignore
            
            return result
            
        except redis.RedisError as e:
            logger.error(f"[REDIS] Failed to retrieve cached task {task_id}: {str(e)}")
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"[REDIS] JSON deserialization error for task {task_id}: {str(e)}")
            return None
            
        except Exception as e:
            logger.error(f"[REDIS] Unexpected error retrieving task {task_id}: {str(e)}")
            return None
    
    def invalidate_cache(self, task_id: str) -> bool:
        """
        Manually invalidate (delete) a cached task result.
        
        Args:
            task_id: Unique task identifier
            
        Returns:
            True if key was deleted, False otherwise
        """
        if not self.redis_available or not self.client:
            return False
        
        try:
            cache_key = f"task:{task_id}"
            deleted = self.client.delete(cache_key)
            
            if deleted:
                logger.info(f"[REDIS] ✓ Invalidated cache: {cache_key}")
                return True
            else:
                logger.debug(f"[REDIS] No cache entry to invalidate: {cache_key}")
                return False
                
        except Exception as e:
            logger.error(f"[REDIS] Error invalidating cache for task {task_id}: {str(e)}")
            return False
    
    def get_all_cached_tasks(self, pattern: str = "task:*") -> list[str]:
        """
        Get list of all cached task IDs matching a pattern.
        
        Args:
            pattern: Redis key pattern (default: "task:*")
            
        Returns:
            List of task IDs (without "task:" prefix)
        """
        if not self.redis_available or not self.client:
            return []
        
        try:
            keys = self.client.keys(pattern)  # type: ignore
            # Remove "task:" prefix from each key
            task_ids = [key.replace("task:", "") for key in keys]  # type: ignore
            
            logger.debug(f"[REDIS] Found {len(task_ids)} cached tasks matching '{pattern}'")
            return task_ids
            
        except Exception as e:
            logger.error(f"[REDIS] Error listing cached tasks: {str(e)}")
            return []
    
    def clear_all_cache(self, pattern: str = "task:*") -> int:
        """
        Clear all cached tasks matching a pattern.
        
        WARNING: Use with caution in production!
        
        Args:
            pattern: Redis key pattern (default: "task:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.redis_available or not self.client:
            return 0
        
        try:
            keys = self.client.keys(pattern)  # type: ignore
            if not keys:
                logger.debug(f"[REDIS] No keys found matching '{pattern}'")
                return 0
            
            deleted = self.client.delete(*keys)  # type: ignore
            logger.warning(f"[REDIS] ⚠ Cleared {deleted} cached tasks matching '{pattern}'")
            
            return int(deleted)  # type: ignore
            
        except Exception as e:
            logger.error(f"[REDIS] Error clearing cache: {str(e)}")
            return 0
    
    def close(self):
        """
        Close Redis connection and cleanup resources.
        """
        if self.client:
            try:
                self.client.close()
                logger.info("[REDIS] Connection closed")
            except Exception as e:
                logger.error(f"[REDIS] Error closing connection: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
