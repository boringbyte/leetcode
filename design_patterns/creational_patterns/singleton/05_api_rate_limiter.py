import threading
import time
from collections import deque


class RateLimiter:

    _instance = None
    _lock = threading.Lock()
    _initialized = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._requests = {}
            self._limits = {}
            self._initialized = True

    def set_limit(self, user_id: str, max_requests: int, time_window_seconds: int):
        self._limits[user_id] = (max_requests, time_window_seconds)
        self._requests[user_id] = deque()

    def is_allowed(self, user_id: str) -> bool:
        """Check if user is allowed to make a request"""
        if user_id not in self._limits:
            # No limit set, allow by default
            return True

        max_requests, time_window = self._limits[user_id]
        current_time = time.time()

        # Initialize request queue if not exists
        if user_id not in self._requests:
            self._requests[user_id] = deque()

        request_queue = self._requests[user_id]

        # Remove old requests outside the time window
        while request_queue and current_time - request_queue[0] > time_window:
            request_queue.popleft()

        # Check if limit exceeded
        if len(request_queue) >= max_requests:
            print(f"Rate limit exceeded for user {user_id}")
            return False

        # Add current request
        request_queue.append(current_time)
        return True

    def get_remaining_requests(self, user_id: str) -> int | None:
        """Get remaining requests for user"""
        if user_id not in self._limits:
            return None

        max_requests, time_window = self._limits[user_id]
        current_time = time.time()

        if user_id not in self._requests:
            return max_requests

        request_queue = self._requests[user_id]

        # Remove old requests
        while request_queue and current_time - request_queue[0] > time_window:
            request_queue.popleft()

        return max_requests - len(request_queue)


if __name__ == '__main__':
    # Set up rate limiter
    limiter = RateLimiter()
    limiter.set_limit("user123", max_requests=5, time_window_seconds=60)

    # API endpoint 1
    limiter_check = RateLimiter()  # Same instance
    if limiter_check.is_allowed("user123"):
        print("Request 1: Allowed")
    else:
        print("Request 1: Denied")

    # API endpoint 2
    limiter_check = RateLimiter()  # Same instance
    if limiter_check.is_allowed("user123"):
        print("Request 2: Allowed")
    else:
        print("Request 2: Denied")

    # Check remaining
    print(f"Remaining requests: {limiter.get_remaining_requests('user123')}")
