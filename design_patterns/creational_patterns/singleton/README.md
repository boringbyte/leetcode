# Singleton Pattern - Complete Guide

## 📋 Table of Contents
- [What is Singleton?](#what-is-singleton)
- [When to Use](#when-to-use)
- [When NOT to Use](#when-not-to-use)
- [Implementation Methods](#implementation-methods)
- [Real-World Examples](#real-world-examples)
- [Common Pitfalls](#common-pitfalls)
- [Best Practices](#best-practices)

---

## What is Singleton?

**Singleton** is a creational design pattern that ensures a class has **only one instance** and provides a **global point of access** to that instance.

### Key Characteristics:
- ✅ Only one instance exists throughout the application lifecycle
- ✅ Global access point to that instance
- ✅ Instance is created only when first requested (lazy initialization)
- ✅ Prevents multiple instantiations

### Visual Representation:
```
First Call:  DatabaseConnection() → Creates Instance A
Second Call: DatabaseConnection() → Returns Instance A (same)
Third Call:  DatabaseConnection() → Returns Instance A (same)
```

---

## When to Use

### ✅ Perfect Use Cases:

#### 1. **Database Connections**
- Only one connection pool should exist
- Expensive to create multiple connections
- Need to manage shared connection state

#### 2. **Configuration Manager**
- Application settings should be consistent across the app
- Loading config once and sharing it is efficient

#### 3. **Logger**
- All parts of the application should log to the same destination
- Need centralized log management

#### 4. **Cache Manager**
- Single source of truth for cached data
- Avoid cache inconsistencies

#### 5. **Thread Pool / Connection Pool**
- Resource management requires single controller
- Avoid resource exhaustion

#### 6. **Hardware Interface Access**
- Printer, file system, device drivers
- Only one process should control hardware at a time

---

## When NOT to Use

### ❌ Avoid Singleton When:

1. **Testing is Important**
   - Singletons make unit testing difficult
   - Hard to mock or replace in tests
   - Creates hidden dependencies

2. **You Need Multiple Instances**
   - If requirements might change (today one DB, tomorrow multiple DBs)
   - Flexibility is more important than restriction

3. **In Multi-threaded Environments Without Care**
   - Can cause race conditions
   - Requires thread-safe implementation

4. **It Hides Dependencies**
   - Makes code harder to understand
   - Better to use dependency injection

5. **You're Using It as a Global Variable**
   - Singleton ≠ Global variable
   - Don't use it just to avoid passing parameters

---

## Implementation Methods

### Method 1: Using `__new__` (Classic Approach)

```python
class Singleton:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)  # no need to pass *args and **kwargs as they are ignored here
        return cls._instance

# Usage
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True - same instance
```

**Pros:**
- Simple and straightforward
- Works with inheritance

**Cons:**
- Not thread-safe
- Can be bypassed

---

### Method 2: Thread-Safe Singleton (Production-Ready)

```python
import threading

class ThreadSafeSingleton:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

# Usage
s1 = ThreadSafeSingleton()
s2 = ThreadSafeSingleton()
print(s1 is s2)  # True
```

**Pros:**
- Thread-safe
- Uses double-checked locking for performance

**Cons:**
- Slightly more complex
- Small performance overhead

---

### Method 3: Metaclass Approach (Advanced)

```python
import threading

class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
    def __init__(self):
        self.value = None

# Usage
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True
```

**Pros:**
- Cleaner separation of concerns
- Can be reused for multiple singleton classes
- Thread-safe

**Cons:**
- More complex to understand
- Metaclass magic can be confusing

---

### Method 4: Decorator Approach

```python
import threading

def singleton(cls):
    instances = {}
    lock = threading.Lock()
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class MySingleton:
    def __init__(self):
        self.value = None

# Usage
s1 = MySingleton()
s2 = MySingleton()
print(s1 is s2)  # True
```

**Pros:**
- Very clean and reusable
- Easy to apply to any class
- Thread-safe

**Cons:**
- The class is no longer a class (it's a function)
- `isinstance()` won't work as expected

---

### Method 5: Module-Level Singleton (Pythonic Way)

Python modules are singletons by nature!

```python
# config.py
class ConfigManager:
    def __init__(self):
        self.settings = {}
    
    def load_config(self, file_path):
        # Load configuration
        self.settings = {"db_host": "localhost", "db_port": 5432}
    
    def get(self, key):
        return self.settings.get(key)

# Create single instance at module level
config_manager = ConfigManager()
```

```python
# app.py
from config import config_manager

# Everyone uses the same instance
config_manager.load_config("config.json")
print(config_manager.get("db_host"))  # localhost
```

```python
# another_module.py
from config import config_manager

# Same instance as in app.py
print(config_manager.get("db_host"))  # localhost
```

**Pros:**
- Most Pythonic approach
- Simple and clean
- No complex patterns needed

**Cons:**
- Import-time initialization
- Less explicit that it's a singleton

---

## Real-World Examples

### Example 1: Database Connection Manager

```python
import threading
import sqlite3

class DatabaseConnection:
    _instance = None
    _lock = threading.Lock()
    _connection = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def connect(self, db_name):
        """Initialize connection only once"""
        if self._connection is None:
            print(f"Creating new database connection to {db_name}")
            self._connection = sqlite3.connect(db_name, check_same_thread=False)
        return self._connection
    
    def execute_query(self, query):
        if self._connection is None:
            raise Exception("Database not connected. Call connect() first.")
        cursor = self._connection.cursor()
        cursor.execute(query)
        return cursor.fetchall()
    
    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None
            print("Database connection closed")

# Usage
db1 = DatabaseConnection()
db1.connect("myapp.db")

db2 = DatabaseConnection()
# No new connection created - uses existing one
print(db1 is db2)  # True

# Both use the same connection
results = db1.execute_query("SELECT * FROM users")
# db2 would use the same connection
```

---

### Example 2: Application Logger

```python
import threading
from datetime import datetime

class Logger:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Ensure __init__ runs only once
        if not self._initialized:
            self.log_file = "app.log"
            self.logs = []
            self._initialized = True
    
    def _write_to_file(self, message):
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def info(self, message):
        log_entry = f"[INFO] [{datetime.now()}] {message}"
        self.logs.append(log_entry)
        self._write_to_file(log_entry)
        print(log_entry)
    
    def error(self, message):
        log_entry = f"[ERROR] [{datetime.now()}] {message}"
        self.logs.append(log_entry)
        self._write_to_file(log_entry)
        print(log_entry)
    
    def warning(self, message):
        log_entry = f"[WARNING] [{datetime.now()}] {message}"
        self.logs.append(log_entry)
        self._write_to_file(log_entry)
        print(log_entry)
    
    def get_all_logs(self):
        return self.logs

# Usage in different modules
# module1.py
logger = Logger()
logger.info("Application started")

# module2.py
logger = Logger()  # Same instance
logger.error("An error occurred")

# module3.py
logger = Logger()  # Same instance
logger.warning("Low memory warning")

# All logs are in the same place
print(f"Total logs: {len(logger.get_all_logs())}")  # 3
```

---

### Example 3: Configuration Manager

```python
import json
import threading

class ConfigManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.config = {}
            self._initialized = True
    
    def load_from_file(self, file_path):
        """Load configuration from JSON file"""
        try:
            with open(file_path, 'r') as f:
                self.config = json.load(f)
            print(f"Configuration loaded from {file_path}")
        except FileNotFoundError:
            print(f"Config file {file_path} not found. Using defaults.")
            self.config = self._get_default_config()
    
    def _get_default_config(self):
        return {
            "app_name": "MyApp",
            "version": "1.0.0",
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "mydb"
            },
            "api": {
                "timeout": 30,
                "retries": 3
            }
        }
    
    def get(self, key, default=None):
        """Get configuration value by key (supports nested keys with dot notation)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def set(self, key, value):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_all(self):
        """Get entire configuration"""
        return self.config

# Usage across application

# app.py
config = ConfigManager()
config.load_from_file("config.json")

# database.py
config = ConfigManager()  # Same instance
db_host = config.get("database.host")
db_port = config.get("database.port")
print(f"Connecting to {db_host}:{db_port}")

# api_client.py
config = ConfigManager()  # Same instance
timeout = config.get("api.timeout")
retries = config.get("api.retries")
print(f"API timeout: {timeout}s, retries: {retries}")

# admin.py
config = ConfigManager()  # Same instance
config.set("api.timeout", 60)  # Update globally
print(config.get("api.timeout"))  # 60
```

---

### Example 4: Cache Manager

```python
import threading
from typing import Any, Optional
from datetime import datetime, timedelta

class CacheManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._cache = {}
            self._expiry = {}
            self._initialized = True
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set a value in cache with optional TTL (time to live)"""
        self._cache[key] = value
        
        if ttl_seconds:
            expiry_time = datetime.now() + timedelta(seconds=ttl_seconds)
            self._expiry[key] = expiry_time
        else:
            self._expiry[key] = None
        
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

# Usage

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
```

---

### Example 5: API Rate Limiter

```python
import threading
import time
from collections import deque
from typing import Optional

class RateLimiter:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # Track requests per user
            self._requests = {}  # user_id -> deque of timestamps
            self._limits = {}    # user_id -> (max_requests, time_window)
            self._initialized = True
    
    def set_limit(self, user_id: str, max_requests: int, time_window_seconds: int):
        """Set rate limit for a user"""
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
    
    def get_remaining_requests(self, user_id: str) -> Optional[int]:
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

# Usage

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
```

---

## Common Pitfalls

### ❌ Pitfall 1: Not Thread-Safe

```python
# BAD - Race condition possible
class BadSingleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)  # Two threads can reach here!
        return cls._instance
```

**Solution:** Use locks (as shown in thread-safe examples above)

---

### ❌ Pitfall 2: `__init__` Called Multiple Times

```python
class BadSingleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.value = 0  # This runs every time!

# Problem
s1 = BadSingleton()
s1.value = 100

s2 = BadSingleton()  # __init__ runs again, resets value!
print(s1.value)  # 0 (not 100!)
```

**Solution:** Use initialization flag

```python
class GoodSingleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.value = 0
            self._initialized = True
```

---

### ❌ Pitfall 3: Singleton Inheritance Problems

```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

class Child1(Singleton):
    pass

class Child2(Singleton):
    pass

# Problem - All share same instance!
c1 = Child1()
c2 = Child2()
print(c1 is c2)  # True - BAD!
```

**Solution:** Use class-specific instances

```python
class Singleton:
    _instances = {}
    
    def __new__(cls):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]
```

---

## Best Practices

### ✅ 1. Use Module-Level Singleton for Simple Cases

```python
# config.py
class Config:
    def __init__(self):
        self.settings = {}

config = Config()  # Single instance at module level
```

This is the most Pythonic way!

---

### ✅ 2. Make It Thread-Safe for Production

Always use locks in multi-threaded environments.

---

### ✅ 3. Document That It's a Singleton

```python
class DatabaseConnection:
    """
    Singleton class for managing database connections.
    
    Only one instance will exist throughout the application.
    Use get_instance() to access the connection.
    """
    pass
```

---

### ✅ 4. Consider Dependency Injection Instead

For better testability:

```python
# Instead of Singleton
class UserService:
    def __init__(self, db_connection):  # Inject dependency
        self.db = db_connection
```

---

### ✅ 5. Provide a Reset Method for Testing

```python
class Singleton:
    _instance = None
    
    @classmethod
    def reset(cls):
        """Reset singleton instance (useful for testing)"""
        cls._instance = None
```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Purpose** | Ensure only one instance exists |
| **Use When** | Database connections, loggers, config, cache |
| **Avoid When** | Testing is critical, need flexibility |
| **Best Implementation** | Module-level for simple, Metaclass for reusable |
| **Key Consideration** | Thread safety in concurrent environments |

