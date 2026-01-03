"""
Global Rate Limiter for Semantic Scholar API

Provides a thread-safe singleton rate limiter that enforces:
- Maximum 1 request per second globally (within this process)
- Shared exponential backoff on 429 responses
- Thread-safe access from multiple workers

Usage:
    from utils.s2_rate_limiter import get_s2_limiter
    
    limiter = get_s2_limiter()
    limiter.wait_for_slot()  # Blocks until rate limit allows
    response = requests.get(s2_url)
    if response.status_code == 429:
        limiter.record_throttle()
"""

import os
import threading
import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class S2RateLimiter:
    """
    Thread-safe global rate limiter for Semantic Scholar API.
    
    Enforces a minimum interval between requests and applies
    exponential backoff when throttling is detected.
    """
    
    def __init__(self, min_interval: float = 1.0, max_backoff: float = 60.0):
        """
        Initialize the rate limiter.
        
        Args:
            min_interval: Minimum seconds between requests (default: 1.0)
            max_backoff: Maximum backoff time in seconds (default: 60.0)
        """
        self._lock = threading.Lock()
        self._last_request_time = 0.0
        self._min_interval = min_interval
        self._max_backoff = max_backoff
        
        # Backoff state
        self._current_backoff = 0.0
        self._backoff_factor = 2.0
        self._consecutive_throttles = 0
        self._last_throttle_time = 0.0
        
        # Stats for monitoring
        self._total_requests = 0
        self._total_throttles = 0
        self._total_wait_time = 0.0
    
    def wait_for_slot(self) -> float:
        """
        Wait until a request slot is available.
        Blocks other threads until this request's slot is complete.
        
        Returns:
            The number of seconds waited.
        """
        with self._lock:
            now = time.time()
            
            # Calculate required wait time
            # Base interval + any active backoff
            effective_interval = self._min_interval + self._current_backoff
            time_since_last = now - self._last_request_time
            wait_time = max(0.0, effective_interval - time_since_last)
            
            if wait_time > 0:
                logger.debug(f"S2 Rate Limiter: Waiting {wait_time:.2f}s (backoff: {self._current_backoff:.2f}s)")
                # IMPORTANT: Keep lock held during sleep to serialize all requests
                time.sleep(wait_time)
                self._total_wait_time += wait_time
            
            # Update state - mark this as the last request time
            self._last_request_time = time.time()
            self._total_requests += 1
            
            # Decay backoff if no throttles for a while (30 seconds per halving)
            # Apply multiple decay steps based on how much time has elapsed
            if self._current_backoff > 0 and self._last_throttle_time > 0:
                elapsed = now - self._last_throttle_time
                if elapsed > 30:
                    # Calculate how many 30-second decay periods have passed
                    decay_periods = int(elapsed / 30)
                    old_backoff = self._current_backoff
                    # Apply decay: halve the backoff for each 30-second period
                    self._current_backoff = self._current_backoff / (2 ** decay_periods)
                    # If decayed to tiny value, just set to 0
                    if self._current_backoff < 0.01:
                        self._current_backoff = 0.0
                        self._consecutive_throttles = 0
                    else:
                        self._consecutive_throttles = max(0, self._consecutive_throttles - decay_periods)
                    if old_backoff != self._current_backoff:
                        logger.debug(f"S2 Rate Limiter: Decayed backoff from {old_backoff:.2f}s to {self._current_backoff:.2f}s ({decay_periods} periods)")
            
            return wait_time
    
    def record_throttle(self):
        """
        Record a throttle event (429 response) to increase backoff.
        """
        with self._lock:
            self._total_throttles += 1
            self._last_throttle_time = time.time()
            self._consecutive_throttles += 1
            
            # Exponential backoff: 1s, 2s, 4s, 8s, ... up to max
            # Use threshold comparison to handle floating point decay artifacts
            if self._current_backoff < 0.01:
                self._current_backoff = 1.0
            else:
                self._current_backoff = min(
                    self._current_backoff * self._backoff_factor,
                    self._max_backoff
                )
            
            logger.warning(
                f"S2 Rate Limiter: Throttled! Backoff now {self._current_backoff:.1f}s "
                f"(consecutive: {self._consecutive_throttles}, total: {self._total_throttles})"
            )
    
    def record_success(self):
        """
        Record a successful request to help decay backoff faster.
        """
        with self._lock:
            # On success, reduce consecutive throttle count
            if self._consecutive_throttles > 0:
                self._consecutive_throttles -= 1
                # Gradually reduce backoff on success
                if self._consecutive_throttles == 0:
                    self._current_backoff = max(0.0, self._current_backoff / 2)
                    logger.debug(f"S2 Rate Limiter: Success, reduced backoff to {self._current_backoff:.2f}s")
    
    @contextmanager
    def acquire(self):
        """
        Context manager that waits for a rate limit slot.
        
        Usage:
            with limiter.acquire():
                response = requests.get(url)
        """
        self.wait_for_slot()
        yield
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                'total_requests': self._total_requests,
                'total_throttles': self._total_throttles,
                'total_wait_time': self._total_wait_time,
                'current_backoff': self._current_backoff,
                'consecutive_throttles': self._consecutive_throttles
            }


# Singleton instance
_limiter_instance = None
_limiter_lock = threading.Lock()


def get_s2_limiter() -> S2RateLimiter:
    """
    Get the global S2 rate limiter singleton.
    
    Returns:
        The shared S2RateLimiter instance.
    """
    global _limiter_instance
    if _limiter_instance is None:
        with _limiter_lock:
            if _limiter_instance is None:
                # Check for API key in environment
                api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
                
                # Determine interval based on key presence
                # With Key: 2.0s (Very safe buffer, officially 10-100/sec but 1.0s was failing)
                # No Key: 3.0s (Public API is strict 1/sec, 3.0s is safe buffer)
                interval = 2.0 if api_key else 3.0
                
                _limiter_instance = S2RateLimiter(min_interval=interval)
                logger.info(f"S2 Rate Limiter: Initialized global instance ({interval}s/req)")
    return _limiter_instance


def reset_s2_limiter():
    """
    Reset the global rate limiter (mainly for testing).
    """
    global _limiter_instance
    with _limiter_lock:
        _limiter_instance = None
