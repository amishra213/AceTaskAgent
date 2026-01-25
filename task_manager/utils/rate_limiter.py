"""
Rate Limiter - Global rate limiting for LLM API calls.

Provides thread-safe rate limiting with configurable:
- Requests per minute (RPM)
- Requests per second (RPS)
- Minimum delay between requests

Usage:
    from task_manager.utils.rate_limiter import global_rate_limiter
    
    # Wait before making LLM call (blocks if rate limit exceeded)
    global_rate_limiter.wait()
    
    # Or configure from environment
    global_rate_limiter.configure_from_env()
"""

import os
import time
import threading
from typing import Optional
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterConfig:
    """Configuration for rate limiter."""
    requests_per_minute: int = 60
    requests_per_second: int = 0
    min_request_delay: float = 0.5


class RateLimiter:
    """
    Thread-safe rate limiter for LLM API calls.
    
    Implements a sliding window rate limiter that tracks request timestamps
    and enforces rate limits across all LLM calls in the application.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_second: int = 0,
        min_request_delay: float = 0.5,
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Max requests per minute (0 = unlimited)
            requests_per_second: Max requests per second (0 = unlimited, overrides RPM if set)
            min_request_delay: Minimum seconds between requests (0 = no delay)
        """
        self._lock = threading.Lock()
        self._request_times: deque = deque()
        self._last_request_time: float = 0
        
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self.min_request_delay = min_request_delay
        
        # Calculate effective rate limit
        if requests_per_second > 0:
            self._window_seconds = 1.0
            self._max_requests = requests_per_second
        elif requests_per_minute > 0:
            self._window_seconds = 60.0
            self._max_requests = requests_per_minute
        else:
            self._window_seconds = 0
            self._max_requests = 0
        
        logger.debug(
            f"Rate limiter initialized: RPM={requests_per_minute}, "
            f"RPS={requests_per_second}, min_delay={min_request_delay}s"
        )
    
    def configure(
        self,
        requests_per_minute: Optional[int] = None,
        requests_per_second: Optional[int] = None,
        min_request_delay: Optional[float] = None,
    ) -> None:
        """
        Reconfigure rate limiter settings.
        
        Args:
            requests_per_minute: Max requests per minute (0 = unlimited)
            requests_per_second: Max requests per second (0 = unlimited)
            min_request_delay: Minimum seconds between requests
        """
        with self._lock:
            if requests_per_minute is not None:
                self.requests_per_minute = requests_per_minute
            if requests_per_second is not None:
                self.requests_per_second = requests_per_second
            if min_request_delay is not None:
                self.min_request_delay = min_request_delay
            
            # Recalculate effective rate limit
            if self.requests_per_second > 0:
                self._window_seconds = 1.0
                self._max_requests = self.requests_per_second
            elif self.requests_per_minute > 0:
                self._window_seconds = 60.0
                self._max_requests = self.requests_per_minute
            else:
                self._window_seconds = 0
                self._max_requests = 0
            
            logger.info(
                f"Rate limiter reconfigured: RPM={self.requests_per_minute}, "
                f"RPS={self.requests_per_second}, min_delay={self.min_request_delay}s"
            )
    
    def configure_from_env(self) -> None:
        """Configure rate limiter from environment variables."""
        rpm = int(os.getenv('LLM_RATE_LIMIT_RPM', '60'))
        rps = int(os.getenv('LLM_RATE_LIMIT_RPS', '0'))
        min_delay = float(os.getenv('LLM_MIN_REQUEST_DELAY', '0.5'))
        
        self.configure(
            requests_per_minute=rpm,
            requests_per_second=rps,
            min_request_delay=min_delay,
        )
    
    def _cleanup_old_requests(self, current_time: float) -> None:
        """Remove request timestamps outside the current window."""
        if self._window_seconds <= 0:
            return
        
        cutoff = current_time - self._window_seconds
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()
    
    def wait(self) -> float:
        """
        Wait until a request can be made within rate limits.
        
        Returns:
            Actual wait time in seconds (0 if no wait needed)
        """
        with self._lock:
            current_time = time.time()
            total_wait = 0.0
            
            # Enforce minimum delay between requests
            if self.min_request_delay > 0 and self._last_request_time > 0:
                time_since_last = current_time - self._last_request_time
                if time_since_last < self.min_request_delay:
                    delay_wait = self.min_request_delay - time_since_last
                    total_wait += delay_wait
                    current_time += delay_wait
            
            # Enforce sliding window rate limit
            if self._max_requests > 0:
                self._cleanup_old_requests(current_time)
                
                while len(self._request_times) >= self._max_requests:
                    # Need to wait until oldest request expires
                    oldest = self._request_times[0]
                    wait_until = oldest + self._window_seconds
                    window_wait = wait_until - current_time
                    
                    if window_wait > 0:
                        total_wait += window_wait
                        current_time += window_wait
                    
                    self._cleanup_old_requests(current_time)
            
            # Actually sleep if needed
            if total_wait > 0:
                logger.debug(f"Rate limiter: waiting {total_wait:.2f}s")
                # Release lock while sleeping
                self._lock.release()
                try:
                    time.sleep(total_wait)
                finally:
                    self._lock.acquire()
                current_time = time.time()
            
            # Record this request
            self._request_times.append(current_time)
            self._last_request_time = current_time
            
            return total_wait
    
    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.
        
        Returns:
            Dictionary with stats including requests in current window
        """
        with self._lock:
            current_time = time.time()
            self._cleanup_old_requests(current_time)
            
            return {
                "requests_in_window": len(self._request_times),
                "max_requests": self._max_requests,
                "window_seconds": self._window_seconds,
                "min_request_delay": self.min_request_delay,
                "requests_per_minute": self.requests_per_minute,
                "requests_per_second": self.requests_per_second,
            }
    
    def reset(self) -> None:
        """Reset rate limiter state (clear all tracked requests)."""
        with self._lock:
            self._request_times.clear()
            self._last_request_time = 0
            logger.debug("Rate limiter reset")


# Global rate limiter instance - shared across all LLM calls
global_rate_limiter = RateLimiter()

# Try to auto-configure from environment on import
# This will use defaults if environment variables aren't set yet
try:
    # Try to load .env if not already loaded
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not available, skip
    
    global_rate_limiter.configure_from_env()
    logger.debug("Rate limiter auto-configured from environment on import")
except Exception as e:
    logger.debug(f"Rate limiter using defaults (env not loaded yet): {e}")
