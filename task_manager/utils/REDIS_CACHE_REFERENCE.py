"""
Redis Cache Manager - Quick Reference

INSTALLATION:
    pip install redis
    redis-server  # Start Redis locally

BASIC USAGE:
    from task_manager.utils import RedisCacheManager
    
    cache = RedisCacheManager()  # Connects to localhost:6379
    
    # Cache a result
    cache.cache_task_result(
        task_id="task_1.2",
        input_data={"query": "search term"},
        output_data={"results": [...]},
        agent_type="web_search_agent"
    )
    
    # Retrieve cached result
    cached = cache.get_cached_result("task_1.2")
    if cached:
        print(cached['output'])  # {"results": [...]}

DATA STRUCTURE:
    Key: task:{task_id}
    Type: Redis Hash
    Fields:
        - input: JSON string of input parameters
        - output: JSON string of execution results
        - timestamp: ISO format creation time
        - agent_type: Agent that created this result
    TTL: 86400 seconds (24 hours)

API METHODS:
    cache_task_result(task_id, input_data, output_data, agent_type=None, ttl=None)
        → Returns: bool (True if cached successfully)
    
    get_cached_result(task_id)
        → Returns: dict with keys [input, output, timestamp, agent_type, ttl]
        → Returns: None if not found or Redis unavailable
    
    invalidate_cache(task_id)
        → Returns: bool (True if deleted)
    
    get_all_cached_tasks(pattern="task:*")
        → Returns: list[str] of task IDs
    
    clear_all_cache(pattern="task:*")
        → Returns: int (number of keys deleted)

CONFIGURATION:
    cache = RedisCacheManager(
        host='localhost',      # Redis host
        port=6379,            # Redis port
        db=0,                 # Database number (0-15)
        password=None,        # Auth password
        default_ttl=86400     # Default TTL in seconds
    )

CONTEXT MANAGER:
    with RedisCacheManager() as cache:
        cache.cache_task_result(...)
    # Connection auto-closed

REDIS CLI COMMANDS:
    # View all cached tasks
    redis-cli KEYS "task:*"
    
    # View specific task
    redis-cli HGETALL "task:task_1.2"
    
    # Check TTL
    redis-cli TTL "task:task_1.2"
    
    # Delete specific task
    redis-cli DEL "task:task_1.2"
    
    # Delete all tasks
    redis-cli --scan --pattern "task:*" | xargs redis-cli DEL

GRACEFUL DEGRADATION:
    # If Redis is unavailable, cache operations fail silently:
    cache = RedisCacheManager(host='nonexistent')
    cache.cache_task_result(...)  # Returns False, doesn't crash
    cached = cache.get_cached_result(...)  # Returns None

ERROR HANDLING:
    try:
        cache.cache_task_result(...)
    except Exception as e:
        # Should not raise - all errors handled internally
        pass

INTEGRATION EXAMPLE:
    class TaskManagerAgent:
        def __init__(self):
            self.cache = RedisCacheManager()
        
        def _execute_web_search_task(self, state):
            task_id = state['active_task_id']
            
            # Check cache
            cached = self.cache.get_cached_result(task_id)
            if cached:
                return self._build_cached_response(state, cached)
            
            # Execute task
            result = self.web_search_agent.search(...)
            
            # Cache result
            self.cache.cache_task_result(
                task_id=task_id,
                input_data=task['parameters'],
                output_data=result,
                agent_type='web_search_agent'
            )
            
            return self._build_response(state, result)

DOCKER DEPLOYMENT:
    docker run -d -p 6379:6379 --name redis-cache redis:7-alpine

MONITORING:
    # Cache hit rate
    redis-cli INFO stats | grep keyspace_hits
    
    # Memory usage
    redis-cli INFO memory | grep used_memory_human
    
    # Connected clients
    redis-cli INFO clients

TTL EXAMPLES:
    # 1 hour
    cache.cache_task_result(..., ttl=3600)
    
    # 1 week
    cache.cache_task_result(..., ttl=604800)
    
    # No expiration (not recommended)
    cache.cache_task_result(..., ttl=-1)

BEST PRACTICES:
    ✓ Cache expensive operations (web search, OCR, LLM calls)
    ✓ Use descriptive task_ids
    ✓ Set appropriate TTLs based on data freshness needs
    ✓ Monitor cache hit rates
    ✓ Use context managers for automatic cleanup
    ✗ Don't cache failed task results
    ✗ Don't use cache for real-time data
    ✗ Don't store sensitive data without encryption
"""
