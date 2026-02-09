# Builder Pattern - Complete Guide

## 📋 Table of Contents
- [What is Builder Pattern?](#what-is-builder-pattern)
- [When to Use](#when-to-use)
- [When NOT to Use](#when-not-to-use)
- [Basic Implementation](#basic-implementation)
- [Fluent Interface (Method Chaining)](#fluent-interface-method-chaining)
- [Director Pattern](#director-pattern)
- [Real-World Examples](#real-world-examples)
- [Common Pitfalls](#common-pitfalls)
- [Best Practices](#best-practices)

---

## What is Builder Pattern?

**Builder Pattern** is a creational design pattern that lets you construct complex objects step by step. It separates the construction of a complex object from its representation, allowing the same construction process to create different representations.

### Key Characteristics:
- ✅ Separates object construction from representation
- ✅ Constructs objects step-by-step
- ✅ Allows different representations using same construction code
- ✅ Provides fine control over construction process
- ✅ Makes code readable and maintainable

### The Problem It Solves:

**❌ Without Builder:**
```python
# Constructor with too many parameters - confusing!
pizza = Pizza(
    "large",           # size
    True,              # cheese
    False,             # pepperoni
    True,              # mushrooms
    False,             # olives
    True,              # bacon
    False,             # onions
    True,              # extra_cheese
    "thin"             # crust_type
)
# What does each True/False mean? 🤔
```

**✅ With Builder:**
```python
pizza = (PizzaBuilder()
         .set_size("large")
         .add_cheese()
         .add_mushrooms()
         .add_bacon()
         .extra_cheese()
         .thin_crust()
         .build())
# Clear and readable! 😊
```

---

## When to Use

### ✅ Perfect Use Cases:

#### 1. **Complex Objects with Many Parameters**
- 5+ constructor parameters
- Many optional parameters
- Combination of required and optional parameters

#### 2. **Telescoping Constructor Problem**
```python
# Telescoping constructors - ugly!
class House:
    def __init__(self, rooms):
        self.rooms = rooms
    
    def __init__(self, rooms, windows):
        self.rooms = rooms
        self.windows = windows
    
    def __init__(self, rooms, windows, doors):
        # Can't do this in Python anyway!
        pass
```

#### 3. **Need Different Representations**
- Same building process, different outputs
- Build HTML, Markdown, or JSON from same data
- Create different configurations of same object

#### 4. **Immutable Objects**
- Object should be immutable after creation
- All properties set during construction
- No setters after object is built

#### 5. **Step-by-Step Construction**
- Construction has distinct phases
- Order of steps matters
- Validation needed between steps

---

## When NOT to Use

### ❌ Avoid Builder When:

1. **Simple Objects**
   - Few parameters (< 3-4)
   - All required parameters
   - No complex initialization logic

2. **Mutable Objects Are Fine**
   - Object needs to change after creation
   - Setters are acceptable

3. **One Standard Configuration**
   - No variation in how objects are built
   - No optional components

4. **Adding Unnecessary Complexity**
   - Simple constructor works fine
   - Builder adds boilerplate without benefit

---

## Basic Implementation

### Simple Builder Without Chaining

```python
class Computer:
    """Product - Complex object being built"""
    
    def __init__(self):
        # Components
        self.cpu = None
        self.ram = None
        self.storage = None
        self.gpu = None
        self.os = None
        self.monitor = None
        self.keyboard = None
        self.mouse = None
    
    def __str__(self):
        specs = []
        if self.cpu: specs.append(f"CPU: {self.cpu}")
        if self.ram: specs.append(f"RAM: {self.ram}")
        if self.storage: specs.append(f"Storage: {self.storage}")
        if self.gpu: specs.append(f"GPU: {self.gpu}")
        if self.os: specs.append(f"OS: {self.os}")
        if self.monitor: specs.append(f"Monitor: {self.monitor}")
        if self.keyboard: specs.append(f"Keyboard: {self.keyboard}")
        if self.mouse: specs.append(f"Mouse: {self.mouse}")
        
        return "Computer Specs:\n" + "\n".join(f"  - {spec}" for spec in specs)

class ComputerBuilder:
    """Builder - Constructs the product step by step"""
    
    def __init__(self):
        self.computer = Computer()
    
    def set_cpu(self, cpu: str):
        self.computer.cpu = cpu
    
    def set_ram(self, ram: str):
        self.computer.ram = ram
    
    def set_storage(self, storage: str):
        self.computer.storage = storage
    
    def set_gpu(self, gpu: str):
        self.computer.gpu = gpu
    
    def set_os(self, os: str):
        self.computer.os = os
    
    def set_monitor(self, monitor: str):
        self.computer.monitor = monitor
    
    def set_keyboard(self, keyboard: str):
        self.computer.keyboard = keyboard
    
    def set_mouse(self, mouse: str):
        self.computer.mouse = mouse
    
    def build(self) -> Computer:
        """Return the constructed product"""
        return self.computer

# Usage
builder = ComputerBuilder()
builder.set_cpu("Intel i9-13900K")
builder.set_ram("32GB DDR5")
builder.set_storage("1TB NVMe SSD")
builder.set_gpu("NVIDIA RTX 4090")
builder.set_os("Windows 11")

computer = builder.build()
print(computer)
```

**Output:**
```
Computer Specs:
  - CPU: Intel i9-13900K
  - RAM: 32GB DDR5
  - Storage: 1TB NVMe SSD
  - GPU: NVIDIA RTX 4090
  - OS: Windows 11
```

---

## Fluent Interface (Method Chaining)

### Builder with Method Chaining - Most Common Pattern

```python
class Pizza:
    """Product - The complex object"""
    
    def __init__(self):
        self.size = None
        self.crust_type = "regular"
        self.toppings = []
        self.cheese = False
        self.sauce = "tomato"
    
    def __str__(self):
        description = f"{self.size} pizza with {self.crust_type} crust"
        if self.cheese:
            description += ", cheese"
        if self.toppings:
            description += f", toppings: {', '.join(self.toppings)}"
        description += f", {self.sauce} sauce"
        return description

class PizzaBuilder:
    """Builder with fluent interface"""
    
    def __init__(self):
        self.pizza = Pizza()
    
    def set_size(self, size: str):
        """Set pizza size"""
        if size not in ["small", "medium", "large", "extra-large"]:
            raise ValueError(f"Invalid size: {size}")
        self.pizza.size = size
        return self  # Return self for chaining!
    
    def set_crust(self, crust_type: str):
        """Set crust type"""
        if crust_type not in ["thin", "regular", "thick", "stuffed"]:
            raise ValueError(f"Invalid crust: {crust_type}")
        self.pizza.crust_type = crust_type
        return self
    
    def add_topping(self, topping: str):
        """Add a single topping"""
        self.pizza.toppings.append(topping)
        return self
    
    def add_toppings(self, *toppings):
        """Add multiple toppings at once"""
        self.pizza.toppings.extend(toppings)
        return self
    
    def add_cheese(self):
        """Add cheese"""
        self.pizza.cheese = True
        return self
    
    def set_sauce(self, sauce: str):
        """Set sauce type"""
        self.pizza.sauce = sauce
        return self
    
    def build(self) -> Pizza:
        """Build and return the pizza"""
        if self.pizza.size is None:
            raise ValueError("Pizza size is required!")
        return self.pizza
    
    def reset(self):
        """Reset builder to create a new pizza"""
        self.pizza = Pizza()
        return self

# Usage - Beautiful fluent interface!
pizza1 = (PizzaBuilder()
          .set_size("large")
          .set_crust("thin")
          .add_cheese()
          .add_toppings("pepperoni", "mushrooms", "olives")
          .set_sauce("marinara")
          .build())

print(pizza1)
# Output: large pizza with thin crust, cheese, toppings: pepperoni, mushrooms, olives, marinara sauce

# Build another pizza with same builder
pizza2 = (PizzaBuilder()
          .set_size("medium")
          .add_cheese()
          .add_topping("bacon")
          .add_topping("onions")
          .build())

print(pizza2)
# Output: medium pizza with regular crust, cheese, toppings: bacon, onions, tomato sauce
```

---

## Director Pattern

The **Director** class defines the order in which to execute building steps, while the **Builder** provides the implementation for those steps.

```python
from abc import ABC, abstractmethod
from typing import List

# ============ PRODUCT ============

class Car:
    """Complex product"""
    
    def __init__(self):
        self.parts: List[str] = []
    
    def add_part(self, part: str):
        self.parts.append(part)
    
    def list_parts(self):
        return f"Car parts: {', '.join(self.parts)}"

# ============ BUILDER INTERFACE ============

class CarBuilder(ABC):
    """Abstract builder interface"""
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def build_engine(self):
        pass
    
    @abstractmethod
    def build_wheels(self):
        pass
    
    @abstractmethod
    def build_doors(self):
        pass
    
    @abstractmethod
    def build_seats(self):
        pass
    
    @abstractmethod
    def build_electronics(self):
        pass
    
    @abstractmethod
    def get_result(self) -> Car:
        pass

# ============ CONCRETE BUILDERS ============

class SportsCarBuilder(CarBuilder):
    """Builder for sports cars"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.car = Car()
    
    def build_engine(self):
        self.car.add_part("V8 Engine - 500HP")
    
    def build_wheels(self):
        self.car.add_part("18-inch Racing Wheels")
    
    def build_doors(self):
        self.car.add_part("2 Gull-wing Doors")
    
    def build_seats(self):
        self.car.add_part("2 Racing Seats")
    
    def build_electronics(self):
        self.car.add_part("Advanced Sport Electronics")
    
    def get_result(self) -> Car:
        result = self.car
        self.reset()  # Reset for next build
        return result

class SUVBuilder(CarBuilder):
    """Builder for SUVs"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.car = Car()
    
    def build_engine(self):
        self.car.add_part("V6 Engine - 300HP")
    
    def build_wheels(self):
        self.car.add_part("20-inch All-terrain Wheels")
    
    def build_doors(self):
        self.car.add_part("4 Standard Doors")
    
    def build_seats(self):
        self.car.add_part("7 Comfortable Seats")
    
    def build_electronics(self):
        self.car.add_part("Family-friendly Entertainment System")
    
    def get_result(self) -> Car:
        result = self.car
        self.reset()
        return result

class ElectricCarBuilder(CarBuilder):
    """Builder for electric cars"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.car = Car()
    
    def build_engine(self):
        self.car.add_part("Electric Motor - 400HP")
    
    def build_wheels(self):
        self.car.add_part("19-inch Aerodynamic Wheels")
    
    def build_doors(self):
        self.car.add_part("4 Frameless Doors")
    
    def build_seats(self):
        self.car.add_part("5 Vegan Leather Seats")
    
    def build_electronics(self):
        self.car.add_part("Autopilot and AI Assistant")
    
    def get_result(self) -> Car:
        result = self.car
        self.reset()
        return result

# ============ DIRECTOR ============

class CarDirector:
    """
    Director defines the order of building steps.
    It works with any builder that implements the Builder interface.
    """
    
    def __init__(self):
        self._builder = None
    
    def set_builder(self, builder: CarBuilder):
        """Set which builder to use"""
        self._builder = builder
    
    def build_minimal_car(self):
        """Build minimal viable car"""
        self._builder.build_engine()
        self._builder.build_wheels()
        self._builder.build_doors()
    
    def build_full_featured_car(self):
        """Build fully featured car"""
        self._builder.build_engine()
        self._builder.build_wheels()
        self._builder.build_doors()
        self._builder.build_seats()
        self._builder.build_electronics()
    
    def build_basic_car(self):
        """Build basic car without electronics"""
        self._builder.build_engine()
        self._builder.build_wheels()
        self._builder.build_doors()
        self._builder.build_seats()

# ============ USAGE ============

# Create director
director = CarDirector()

print("=== Building Sports Car (Full Featured) ===")
sports_builder = SportsCarBuilder()
director.set_builder(sports_builder)
director.build_full_featured_car()
sports_car = sports_builder.get_result()
print(sports_car.list_parts())

print("\n=== Building SUV (Basic) ===")
suv_builder = SUVBuilder()
director.set_builder(suv_builder)
director.build_basic_car()
suv = suv_builder.get_result()
print(suv.list_parts())

print("\n=== Building Electric Car (Minimal) ===")
electric_builder = ElectricCarBuilder()
director.set_builder(electric_builder)
director.build_minimal_car()
electric_car = electric_builder.get_result()
print(electric_car.list_parts())

print("\n=== Building Without Director ===")
# Can also build manually without director
custom_builder = SportsCarBuilder()
custom_builder.build_engine()
custom_builder.build_wheels()
custom_builder.build_electronics()  # Custom configuration
custom_car = custom_builder.get_result()
print(custom_car.list_parts())
```

**Output:**
```
=== Building Sports Car (Full Featured) ===
Car parts: V8 Engine - 500HP, 18-inch Racing Wheels, 2 Gull-wing Doors, 2 Racing Seats, Advanced Sport Electronics

=== Building SUV (Basic) ===
Car parts: V6 Engine - 300HP, 20-inch All-terrain Wheels, 4 Standard Doors, 7 Comfortable Seats

=== Building Electric Car (Minimal) ===
Car parts: Electric Motor - 400HP, 19-inch Aerodynamic Wheels, 4 Frameless Doors

=== Building Without Director ===
Car parts: V8 Engine - 500HP, 18-inch Racing Wheels, Advanced Sport Electronics
```

---

## Real-World Examples

### Example 1: HTTP Request Builder

```python
from typing import Dict, Optional, Any
from enum import Enum

class HttpMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

class HttpRequest:
    """Product - Complex HTTP request"""
    
    def __init__(self):
        self.method: Optional[HttpMethod] = None
        self.url: Optional[str] = None
        self.headers: Dict[str, str] = {}
        self.query_params: Dict[str, str] = {}
        self.body: Optional[Any] = None
        self.timeout: int = 30
        self.follow_redirects: bool = True
        self.verify_ssl: bool = True
    
    def __str__(self):
        lines = [
            f"HTTP Request:",
            f"  Method: {self.method.value if self.method else 'Not set'}",
            f"  URL: {self.url}",
        ]
        
        if self.headers:
            lines.append(f"  Headers:")
            for key, value in self.headers.items():
                lines.append(f"    {key}: {value}")
        
        if self.query_params:
            lines.append(f"  Query Params: {self.query_params}")
        
        if self.body:
            lines.append(f"  Body: {self.body}")
        
        lines.append(f"  Timeout: {self.timeout}s")
        lines.append(f"  Follow Redirects: {self.follow_redirects}")
        lines.append(f"  Verify SSL: {self.verify_ssl}")
        
        return "\n".join(lines)

class HttpRequestBuilder:
    """Builder for HTTP requests with fluent interface"""
    
    def __init__(self):
        self.request = HttpRequest()
    
    def method(self, method: HttpMethod):
        """Set HTTP method"""
        self.request.method = method
        return self
    
    def get(self, url: str):
        """Shortcut for GET request"""
        self.request.method = HttpMethod.GET
        self.request.url = url
        return self
    
    def post(self, url: str):
        """Shortcut for POST request"""
        self.request.method = HttpMethod.POST
        self.request.url = url
        return self
    
    def put(self, url: str):
        """Shortcut for PUT request"""
        self.request.method = HttpMethod.PUT
        self.request.url = url
        return self
    
    def delete(self, url: str):
        """Shortcut for DELETE request"""
        self.request.method = HttpMethod.DELETE
        self.request.url = url
        return self
    
    def url(self, url: str):
        """Set URL"""
        self.request.url = url
        return self
    
    def header(self, key: str, value: str):
        """Add a header"""
        self.request.headers[key] = value
        return self
    
    def headers(self, headers: Dict[str, str]):
        """Add multiple headers"""
        self.request.headers.update(headers)
        return self
    
    def auth_bearer(self, token: str):
        """Add Bearer token authentication"""
        self.request.headers['Authorization'] = f'Bearer {token}'
        return self
    
    def auth_basic(self, username: str, password: str):
        """Add Basic authentication"""
        import base64
        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        self.request.headers['Authorization'] = f'Basic {credentials}'
        return self
    
    def content_type(self, content_type: str):
        """Set Content-Type header"""
        self.request.headers['Content-Type'] = content_type
        return self
    
    def json_content(self):
        """Set Content-Type to application/json"""
        return self.content_type('application/json')
    
    def query_param(self, key: str, value: str):
        """Add query parameter"""
        self.request.query_params[key] = value
        return self
    
    def query_params(self, params: Dict[str, str]):
        """Add multiple query parameters"""
        self.request.query_params.update(params)
        return self
    
    def body(self, body: Any):
        """Set request body"""
        self.request.body = body
        return self
    
    def json_body(self, data: Dict):
        """Set JSON body and content type"""
        self.request.body = data
        self.json_content()
        return self
    
    def timeout(self, seconds: int):
        """Set timeout"""
        self.request.timeout = seconds
        return self
    
    def no_redirects(self):
        """Disable following redirects"""
        self.request.follow_redirects = False
        return self
    
    def no_ssl_verify(self):
        """Disable SSL verification (use cautiously!)"""
        self.request.verify_ssl = False
        return self
    
    def build(self) -> HttpRequest:
        """Build and return the request"""
        if self.request.method is None:
            raise ValueError("HTTP method is required")
        if self.request.url is None:
            raise ValueError("URL is required")
        return self.request
    
    def reset(self):
        """Reset builder"""
        self.request = HttpRequest()
        return self

# ============ USAGE ============

# Example 1: Simple GET request
request1 = (HttpRequestBuilder()
            .get("https://api.example.com/users")
            .header("Accept", "application/json")
            .query_param("page", "1")
            .query_param("limit", "10")
            .build())

print("=== GET Request ===")
print(request1)

# Example 2: POST request with JSON body
request2 = (HttpRequestBuilder()
            .post("https://api.example.com/users")
            .auth_bearer("eyJhbGciOiJIUzI1...")
            .json_body({
                "name": "John Doe",
                "email": "john@example.com",
                "age": 30
            })
            .timeout(60)
            .build())

print("\n=== POST Request ===")
print(request2)

# Example 3: Complex request with multiple settings
request3 = (HttpRequestBuilder()
            .put("https://api.example.com/users/123")
            .headers({
                "Accept": "application/json",
                "User-Agent": "MyApp/1.0"
            })
            .auth_basic("admin", "password123")
            .json_body({"status": "active"})
            .query_params({"notify": "true", "async": "false"})
            .timeout(120)
            .no_redirects()
            .build())

print("\n=== PUT Request ===")
print(request3)

# Example 4: DELETE request
request4 = (HttpRequestBuilder()
            .delete("https://api.example.com/users/123")
            .auth_bearer("token123")
            .build())

print("\n=== DELETE Request ===")
print(request4)
```

---

### Example 2: SQL Query Builder

```python
from typing import List, Optional, Dict, Any
from enum import Enum

class JoinType(Enum):
    INNER = "INNER JOIN"
    LEFT = "LEFT JOIN"
    RIGHT = "RIGHT JOIN"
    FULL = "FULL JOIN"

class SQLQuery:
    """Product - SQL Query"""
    
    def __init__(self):
        self.query_type: Optional[str] = None
        self.table: Optional[str] = None
        self.columns: List[str] = []
        self.where_conditions: List[str] = []
        self.joins: List[tuple] = []
        self.group_by: List[str] = []
        self.having: Optional[str] = None
        self.order_by: List[tuple] = []
        self.limit: Optional[int] = None
        self.offset: Optional[int] = None
    
    def to_sql(self) -> str:
        """Generate SQL string"""
        if self.query_type == "SELECT":
            return self._build_select()
        elif self.query_type == "INSERT":
            return self._build_insert()
        elif self.query_type == "UPDATE":
            return self._build_update()
        elif self.query_type == "DELETE":
            return self._build_delete()
        else:
            raise ValueError("Query type not set")
    
    def _build_select(self) -> str:
        # SELECT clause
        cols = ", ".join(self.columns) if self.columns else "*"
        query = f"SELECT {cols}"
        
        # FROM clause
        query += f"\nFROM {self.table}"
        
        # JOINs
        for join_type, join_table, join_condition in self.joins:
            query += f"\n{join_type.value} {join_table} ON {join_condition}"
        
        # WHERE clause
        if self.where_conditions:
            query += f"\nWHERE {' AND '.join(self.where_conditions)}"
        
        # GROUP BY
        if self.group_by:
            query += f"\nGROUP BY {', '.join(self.group_by)}"
        
        # HAVING
        if self.having:
            query += f"\nHAVING {self.having}"
        
        # ORDER BY
        if self.order_by:
            order_parts = [f"{col} {direction}" for col, direction in self.order_by]
            query += f"\nORDER BY {', '.join(order_parts)}"
        
        # LIMIT
        if self.limit:
            query += f"\nLIMIT {self.limit}"
        
        # OFFSET
        if self.offset:
            query += f"\nOFFSET {self.offset}"
        
        return query + ";"
    
    def _build_insert(self) -> str:
        cols = ", ".join(self.columns)
        placeholders = ", ".join(["?" for _ in self.columns])
        return f"INSERT INTO {self.table} ({cols})\nVALUES ({placeholders});"
    
    def _build_update(self) -> str:
        set_clause = ", ".join([f"{col} = ?" for col in self.columns])
        query = f"UPDATE {self.table}\nSET {set_clause}"
        
        if self.where_conditions:
            query += f"\nWHERE {' AND '.join(self.where_conditions)}"
        
        return query + ";"
    
    def _build_delete(self) -> str:
        query = f"DELETE FROM {self.table}"
        
        if self.where_conditions:
            query += f"\nWHERE {' AND '.join(self.where_conditions)}"
        
        return query + ";"

class SQLQueryBuilder:
    """Builder for SQL queries"""
    
    def __init__(self):
        self.query = SQLQuery()
    
    def select(self, *columns: str):
        """Start a SELECT query"""
        self.query.query_type = "SELECT"
        self.query.columns = list(columns) if columns else []
        return self
    
    def insert_into(self, table: str):
        """Start an INSERT query"""
        self.query.query_type = "INSERT"
        self.query.table = table
        return self
    
    def update(self, table: str):
        """Start an UPDATE query"""
        self.query.query_type = "UPDATE"
        self.query.table = table
        return self
    
    def delete_from(self, table: str):
        """Start a DELETE query"""
        self.query.query_type = "DELETE"
        self.query.table = table
        return self
    
    def from_table(self, table: str):
        """Set the table for SELECT"""
        self.query.table = table
        return self
    
    def columns(self, *columns: str):
        """Set columns for INSERT or UPDATE"""
        self.query.columns = list(columns)
        return self
    
    def where(self, condition: str):
        """Add WHERE condition"""
        self.query.where_conditions.append(condition)
        return self
    
    def and_where(self, condition: str):
        """Add another WHERE condition (alias for where)"""
        return self.where(condition)
    
    def join(self, table: str, condition: str, join_type: JoinType = JoinType.INNER):
        """Add JOIN clause"""
        self.query.joins.append((join_type, table, condition))
        return self
    
    def inner_join(self, table: str, condition: str):
        """Add INNER JOIN"""
        return self.join(table, condition, JoinType.INNER)
    
    def left_join(self, table: str, condition: str):
        """Add LEFT JOIN"""
        return self.join(table, condition, JoinType.LEFT)
    
    def right_join(self, table: str, condition: str):
        """Add RIGHT JOIN"""
        return self.join(table, condition, JoinType.RIGHT)
    
    def group_by(self, *columns: str):
        """Add GROUP BY clause"""
        self.query.group_by.extend(columns)
        return self
    
    def having(self, condition: str):
        """Add HAVING clause"""
        self.query.having = condition
        return self
    
    def order_by(self, column: str, direction: str = "ASC"):
        """Add ORDER BY clause"""
        self.query.order_by.append((column, direction))
        return self
    
    def limit(self, count: int):
        """Add LIMIT clause"""
        self.query.limit = count
        return self
    
    def offset(self, count: int):
        """Add OFFSET clause"""
        self.query.offset = count
        return self
    
    def build(self) -> SQLQuery:
        """Build and return the query"""
        return self.query
    
    def to_sql(self) -> str:
        """Build and return SQL string directly"""
        return self.query.to_sql()
    
    def reset(self):
        """Reset builder"""
        self.query = SQLQuery()
        return self

# ============ USAGE ============

# Example 1: Simple SELECT
query1 = (SQLQueryBuilder()
          .select("id", "name", "email")
          .from_table("users")
          .where("age > 18")
          .where("status = 'active'")
          .order_by("name", "ASC")
          .limit(10)
          .to_sql())

print("=== Simple SELECT ===")
print(query1)

# Example 2: SELECT with JOIN
query2 = (SQLQueryBuilder()
          .select("users.name", "orders.total", "orders.created_at")
          .from_table("users")
          .inner_join("orders", "users.id = orders.user_id")
          .where("orders.status = 'completed'")
          .order_by("orders.created_at", "DESC")
          .to_sql())

print("\n=== SELECT with JOIN ===")
print(query2)

# Example 3: Complex SELECT with aggregation
query3 = (SQLQueryBuilder()
          .select("department", "COUNT(*) as employee_count", "AVG(salary) as avg_salary")
          .from_table("employees")
          .where("hire_date > '2020-01-01'")
          .group_by("department")
          .having("COUNT(*) > 5")
          .order_by("avg_salary", "DESC")
          .to_sql())

print("\n=== Aggregation Query ===")
print(query3)

# Example 4: INSERT query
query4 = (SQLQueryBuilder()
          .insert_into("users")
          .columns("name", "email", "age")
          .to_sql())

print("\n=== INSERT Query ===")
print(query4)

# Example 5: UPDATE query
query5 = (SQLQueryBuilder()
          .update("users")
          .columns("status", "updated_at")
          .where("id = 123")
          .to_sql())

print("\n=== UPDATE Query ===")
print(query5)

# Example 6: DELETE query
query6 = (SQLQueryBuilder()
          .delete_from("users")
          .where("last_login < '2020-01-01'")
          .where("status = 'inactive'")
          .to_sql())

print("\n=== DELETE Query ===")
print(query6)

# Example 7: Pagination
query7 = (SQLQueryBuilder()
          .select()
          .from_table("products")
          .where("category = 'electronics'")
          .order_by("price", "DESC")
          .limit(20)
          .offset(40)  # Page 3 (20 per page)
          .to_sql())

print("\n=== Pagination Query ===")
print(query7)
```

---

### Example 3: Email Builder

```python
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Attachment:
    filename: str
    content: bytes
    content_type: str

class Email:
    """Product - Email message"""
    
    def __init__(self):
        self.from_address: Optional[str] = None
        self.to_addresses: List[str] = []
        self.cc_addresses: List[str] = []
        self.bcc_addresses: List[str] = []
        self.subject: Optional[str] = None
        self.body_text: Optional[str] = None
        self.body_html: Optional[str] = None
        self.attachments: List[Attachment] = []
        self.headers: Dict[str, str] = {}
        self.priority: str = "normal"
        self.reply_to: Optional[str] = None
        self.created_at: datetime = datetime.now()
    
    def __str__(self):
        lines = [
            "=" * 60,
            f"From: {self.from_address}",
            f"To: {', '.join(self.to_addresses)}",
        ]
        
        if self.cc_addresses:
            lines.append(f"CC: {', '.join(self.cc_addresses)}")
        
        if self.bcc_addresses:
            lines.append(f"BCC: {', '.join(self.bcc_addresses)}")
        
        if self.reply_to:
            lines.append(f"Reply-To: {self.reply_to}")
        
        lines.append(f"Subject: {self.subject}")
        lines.append(f"Priority: {self.priority}")
        
        if self.attachments:
            attach_names = [a.filename for a in self.attachments]
            lines.append(f"Attachments: {', '.join(attach_names)}")
        
        lines.append("=" * 60)
        
        if self.body_text:
            lines.append("\n[Text Body]")
            lines.append(self.body_text)
        
        if self.body_html:
            lines.append("\n[HTML Body]")
            lines.append(self.body_html)
        
        return "\n".join(lines)

class EmailBuilder:
    """Builder for Email with fluent interface"""
    
    def __init__(self):
        self.email = Email()
    
    def from_address(self, address: str):
        """Set sender address"""
        self.email.from_address = address
        return self
    
    def to(self, *addresses: str):
        """Add recipient addresses"""
        self.email.to_addresses.extend(addresses)
        return self
    
    def cc(self, *addresses: str):
        """Add CC addresses"""
        self.email.cc_addresses.extend(addresses)
        return self
    
    def bcc(self, *addresses: str):
        """Add BCC addresses"""
        self.email.bcc_addresses.extend(addresses)
        return self
    
    def subject(self, subject: str):
        """Set email subject"""
        self.email.subject = subject
        return self
    
    def text_body(self, body: str):
        """Set plain text body"""
        self.email.body_text = body
        return self
    
    def html_body(self, body: str):
        """Set HTML body"""
        self.email.body_html = body
        return self
    
    def attach_file(self, filename: str, content: bytes, content_type: str = "application/octet-stream"):
        """Attach a file"""
        attachment = Attachment(filename, content, content_type)
        self.email.attachments.append(attachment)
        return self
    
    def attach_pdf(self, filename: str, content: bytes):
        """Attach a PDF file"""
        return self.attach_file(filename, content, "application/pdf")
    
    def attach_image(self, filename: str, content: bytes):
        """Attach an image"""
        content_type = "image/jpeg" if filename.endswith(".jpg") else "image/png"
        return self.attach_file(filename, content, content_type)
    
    def header(self, key: str, value: str):
        """Add custom header"""
        self.email.headers[key] = value
        return self
    
    def priority_high(self):
        """Set high priority"""
        self.email.priority = "high"
        return self
    
    def priority_low(self):
        """Set low priority"""
        self.email.priority = "low"
        return self
    
    def reply_to(self, address: str):
        """Set Reply-To address"""
        self.email.reply_to = address
        return self
    
    def build(self) -> Email:
        """Build and return email"""
        # Validation
        if not self.email.from_address:
            raise ValueError("From address is required")
        if not self.email.to_addresses:
            raise ValueError("At least one recipient is required")
        if not self.email.subject:
            raise ValueError("Subject is required")
        if not self.email.body_text and not self.email.body_html:
            raise ValueError("Email body (text or HTML) is required")
        
        return self.email
    
    def reset(self):
        """Reset builder"""
        self.email = Email()
        return self

# ============ USAGE ============

# Example 1: Simple text email
email1 = (EmailBuilder()
          .from_address("sender@example.com")
          .to("recipient@example.com")
          .subject("Hello World")
          .text_body("This is a simple text email.")
          .build())

print("=== Simple Email ===")
print(email1)

# Example 2: Email with HTML and attachments
email2 = (EmailBuilder()
          .from_address("sales@company.com")
          .to("client@example.com", "manager@example.com")
          .cc("team@company.com")
          .subject("Q4 Sales Report")
          .html_body("""
              <html>
                <body>
                  <h1>Q4 Sales Report</h1>
                  <p>Please find attached the Q4 sales report.</p>
                  <p>Best regards,<br>Sales Team</p>
                </body>
              </html>
          """)
          .attach_pdf("Q4_Report.pdf", b"PDF content here")
          .attach_file("data.xlsx", b"Excel content", "application/vnd.ms-excel")
          .priority_high()
          .build())

print("\n=== HTML Email with Attachments ===")
print(email2)

# Example 3: Marketing email
email3 = (EmailBuilder()
          .from_address("marketing@company.com")
          .to("customer@example.com")
          .reply_to("support@company.com")
          .subject("Exclusive Offer Just for You! 🎉")
          .text_body("Visit our website for exclusive deals.")
          .html_body("""
              <html>
                <body style="font-family: Arial;">
                  <h2>Special Offer!</h2>
                  <p>Get 20% off your next purchase.</p>
                  <a href="https://example.com">Shop Now</a>
                </body>
              </html>
          """)
          .header("X-Campaign-ID", "SPRING2024")
          .build())

print("\n=== Marketing Email ===")
print(email3)

# Example 4: Notification email
email4 = (EmailBuilder()
          .from_address("noreply@app.com")
          .to("user@example.com")
          .bcc("admin@app.com")  # BCC admin for monitoring
          .subject("Your password was changed")
          .text_body("""
              Your password was recently changed.
              
              If this wasn't you, please contact support immediately.
              
              Thanks,
              Security Team
          """)
          .priority_high()
          .build())

print("\n=== Notification Email ===")
print(email4)
```

---

## Common Pitfalls

### ❌ Pitfall 1: Forgetting to Return `self` in Builder Methods

```python
# BAD - Breaks method chaining
class BadBuilder:
    def set_name(self, name):
        self.name = name
        # Missing return self!
    
    def set_age(self, age):
        self.age = age
        # Missing return self!

# This won't work:
# builder.set_name("Alice").set_age(30)  # Error!

# GOOD - Returns self
class GoodBuilder:
    def set_name(self, name):
        self.name = name
        return self  # ✅
    
    def set_age(self, age):
        self.age = age
        return self  # ✅

# Now works:
builder.set_name("Alice").set_age(30).build()  # ✅
```

---

### ❌ Pitfall 2: Not Validating Before Building

```python
# BAD - No validation
class BadBuilder:
    def build(self):
        return Product(self.required_field)  # Might be None!

# GOOD - Validate required fields
class GoodBuilder:
    def build(self):
        if self.required_field is None:
            raise ValueError("required_field must be set")
        return Product(self.required_field)
```

---

### ❌ Pitfall 3: Mutable Default Arguments

```python
# BAD - Mutable default
class BadBuilder:
    def __init__(self, items=[]):  # Don't do this!
        self.items = items

# GOOD - Use None
class GoodBuilder:
    def __init__(self, items=None):
        self.items = items if items is not None else []
```

---

### ❌ Pitfall 4: Not Resetting Builder After build()

```python
# BAD - Builder state carries over
class BadBuilder:
    def build(self):
        return self.product  # Same product reused!

# GOOD - Reset after build
class GoodBuilder:
    def build(self):
        result = self.product
        self.product = Product()  # Create new for next build
        return result
```

---

## Best Practices

### ✅ 1. Make Immutable Products

```python
class ImmutableProduct:
    def __init__(self, name, age):
        self._name = name
        self._age = age
    
    @property
    def name(self):
        return self._name
    
    @property
    def age(self):
        return self._age
    
    # No setters - immutable after creation
```

---

### ✅ 2. Provide Sensible Defaults

```python
class Builder:
    def __init__(self):
        self.timeout = 30  # Default
        self.retries = 3   # Default
        self.verify_ssl = True  # Default
```

---

### ✅ 3. Use Type Hints

```python
from typing import List, Optional

class Builder:
    def add_items(self, items: List[str]) -> 'Builder':
        self.items.extend(items)
        return self
    
    def build(self) -> Product:
        return Product(self.items)
```

---

### ✅ 4. Document Complex Builders

```python
class ComplexBuilder:
    """
    Builder for creating complex Product instances.
    
    Example:
        product = (ComplexBuilder()
                   .set_name("Widget")
                   .set_price(99.99)
                   .add_feature("waterproof")
                   .build())
    
    Required fields:
        - name: Product name
        - price: Product price
    
    Optional fields:
        - features: List of features
        - description: Product description
    """
    pass
```

---

### ✅ 5. Consider Using Dataclasses (Python 3.7+)

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Product:
    name: str
    price: float
    features: List[str] = field(default_factory=list)
    
    # Builder pattern built-in!
    @classmethod
    def builder(cls):
        return ProductBuilder()

class ProductBuilder:
    def __init__(self):
        self.name = None
        self.price = None
        self.features = []
    
    def set_name(self, name: str):
        self.name = name
        return self
    
    def set_price(self, price: float):
        self.price = price
        return self
    
    def add_feature(self, feature: str):
        self.features.append(feature)
        return self
    
    def build(self) -> Product:
        return Product(self.name, self.price, self.features)
```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Purpose** | Construct complex objects step-by-step |
| **Use When** | Many parameters, optional fields, immutable objects |
| **Avoid When** | Simple objects, few parameters |
| **Key Feature** | Method chaining (fluent interface) |
| **Benefit** | Readable, flexible, maintainable code |

---

**Comparison with Other Patterns:**

| Pattern | Creates | Flexibility | Complexity |
|---------|---------|-------------|------------|
| **Factory** | Simple objects | Medium | Low |
| **Builder** | Complex objects | High | Medium |
| **Prototype** | Copies | Low | Low |
