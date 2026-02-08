import threading
from typing import Any
from datetime import datetime, timedelta


class CacheManager:

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
            self._cache = {}
            self._expiry = {}
            self._initialized = True

    def set(self, key: str, value: Any, ttl_seconds: int | None = None):
        """Set a value in cache with optional TTL (time to live)"""
        self._cache[key] = value

        if ttl_seconds:
            expiry_time = datetime.now() + timedelta(seconds=ttl_seconds)
            self._expiry[key] = expiry_time
        else:
            self._expiry[key] = value

        print(f"Cache SET: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from cache"""
        # Check if key exists
        if key not in self._cache:
            print(f"Cache MISS: {key}")
            return default

        # Check if expired
        if self._expiry[key] and datetime.now() > self._expiry[key]:
            print(f"Cache EXPIRED: {key}")
            del self._cache[key]
            del self._expiry[key]
            return default

        print(f"Cache HIT: {key}")
        return self._cache[key]

    def delete(self, key: str):
        """Delete a key from cache"""
        if key in self._cache:
            del self._cache[key]
            del self._expiry[key]
            print(f"Cache DELETE: {key}")

    def clear(self):
        """Clear entire cache"""
        self._cache.clear()
        self._expiry.clear()
        print("Cache CLEARED")

    def get_stats(self):
        """Get cache statistics"""
        return {
            "total_keys": len(self._cache),
            "keys": list(self._cache.keys())
        }

if __name__ == '__main__':

    # service1.py - User service
    cache = CacheManager()
    cache.set("user:123", {"name": "Alice", "email": "alice@example.com"}, ttl_seconds=300)

    # service2.py - Product service
    cache = CacheManager()  # Same instance
    user_data = cache.get("user:123")  # Cache HIT
    print(user_data)  # {'name': 'Alice', 'email': 'alice@example.com'}

    # service3.py - Order service
    cache = CacheManager()  # Same instance
    cache.set("order:456", {"total": 99.99, "items": 3})

    print(cache.get_stats())
    # {'total_keys': 2, 'keys': ['user:123', 'order:456']}