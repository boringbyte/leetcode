# Prototype Pattern - Complete Guide

## 📋 Table of Contents
- [What is Prototype Pattern?](#what-is-prototype-pattern)
- [When to Use](#when-to-use)
- [When NOT to Use](#when-not-to-use)
- [Shallow vs Deep Copy](#shallow-vs-deep-copy)
- [Basic Implementation](#basic-implementation)
- [Prototype Registry](#prototype-registry)
- [Real-World Examples](#real-world-examples)
- [Common Pitfalls](#common-pitfalls)
- [Best Practices](#best-practices)

---

## What is Prototype Pattern?

**Prototype Pattern** is a creational design pattern that lets you copy existing objects without making your code dependent on their classes. Instead of creating new objects from scratch, you clone existing objects (prototypes).

### Key Characteristics:
- ✅ Creates new objects by copying existing ones
- ✅ Avoids expensive initialization
- ✅ Hides complexity of creating new instances
- ✅ Reduces subclassing
- ✅ Can create objects at runtime

### Visual Representation:
```
Original Object (Prototype)
       ↓
   clone()
       ↓
New Object (Clone) - Independent copy with same properties
```

### The Problem It Solves:

**Scenario:** You have an object that takes a long time to create (database calls, complex calculations, file loading).

**❌ Without Prototype:**
```python
class ExpensiveObject:
    pass

# Every time you need a similar object, recreate from scratch
obj1 = ExpensiveObject()    # Takes 5 seconds to initialize
obj1.load_from_database()   # Takes 3 seconds
obj1.complex_calculation()  # Takes 2 seconds
# Total: 10 seconds

obj2 = ExpensiveObject()    # Another 10 seconds!
obj2.load_from_database()
obj2.complex_calculation()
```

**✅ With Prototype:**
```python
class ExpensiveObject:
    pass

# Create prototype once
prototype = ExpensiveObject()  # 10 seconds (one time)
prototype.load_from_database()
prototype.complex_calculation()

# Clone it instantly
obj1 = prototype.clone()  # Instant!
obj2 = prototype.clone()  # Instant!
obj3 = prototype.clone()  # Instant!
```

---

## When to Use

### ✅ Perfect Use Cases:

#### 1. **Object Creation is Expensive**
- Complex initialization logic
- Database queries required
- Network calls needed
- Heavy computations
- Large file loading

#### 2. **Need Many Similar Objects**
- Game development (enemies, bullets, particles)
- UI elements (buttons, forms with same style)
- Document templates
- Configuration objects

#### 3. **Object Configuration is Complex**
- Many properties to set
- Complex state setup
- Easier to clone and modify than create from scratch

#### 4. **Reduce Number of Subclasses**
- Instead of creating subclass for each variation
- Clone and modify prototype

#### 5. **Runtime Object Creation**
- Don't know exact types at compile time
- Types determined by user input or configuration
- Dynamic object creation

---

## When NOT to Use

### ❌ Avoid Prototype When:

1. **Simple Object Creation**
   - Object is trivial to create
   - No expensive initialization
   - Constructor is sufficient

2. **Objects Have Complex References**
   - Circular references
   - Deep copy is complicated or impossible

3. **Immutable Objects**
   - If objects never change, no need to clone
   - Can just share the same instance

4. **Cloning is More Expensive Than Creating**
   - Deep copy costs more than initialization
   - Complex object graphs

---

## Shallow vs Deep Copy

Understanding the difference is **crucial** for Prototype Pattern!

### Shallow Copy

Creates a new object but **references** to nested objects are shared.

```python
import copy

class Address:
    def __init__(self, street, city):
        self.street = street
        self.city = city
    
    def __str__(self):
        return f"{self.street}, {self.city}"

class Person:
    def __init__(self, name, age, address):
        self.name = name
        self.age = age
        self.address = address  # Reference to Address object
    
    def shallow_clone(self):
        return copy.copy(self)  # Shallow copy
    
    def __str__(self):
        return f"{self.name}, {self.age}, lives at {self.address}"

# Original
original_address = Address("123 Main St", "Boston")
original_person = Person("Alice", 30, original_address)

# Shallow clone
cloned_person = original_person.shallow_clone()
cloned_person.name = "Bob"  # Change primitive - OK
cloned_person.age = 25

# Change nested object
cloned_person.address.city = "New York"  # ⚠️ AFFECTS ORIGINAL!

print("Original:", original_person)
# Output: Alice, 30, lives at 123 Main St, New York  ← City changed!

print("Clone:", cloned_person)
# Output: Bob, 25, lives at 123 Main St, New York
```

**Result:** Both share the same `Address` object!

```
Original Person → name: "Alice"
                  age: 30
                  address: [Address Object] ← Shared!
                            ↑
Cloned Person →   name: "Bob"
                  age: 25
                  address: ┘ (points to same object)
```

---

### Deep Copy

Creates a new object and **recursively copies** all nested objects.

```python
import copy

class Person:
    def __init__(self, name, age, address):
        self.name = name
        self.age = age
        self.address = address
    
    def deep_clone(self):
        return copy.deepcopy(self)  # Deep copy
    
    def __str__(self):
        return f"{self.name}, {self.age}, lives at {self.address}"

# Original
original_address = Address("123 Main St", "Boston")
original_person = Person("Alice", 30, original_address)

# Deep clone
cloned_person = original_person.deep_clone()
cloned_person.name = "Bob"
cloned_person.age = 25

# Change nested object
cloned_person.address.city = "New York"  # ✅ Only affects clone

print("Original:", original_person)
# Output: Alice, 30, lives at 123 Main St, Boston  ← Unchanged!

print("Clone:", cloned_person)
# Output: Bob, 25, lives at 123 Main St, New York
```

**Result:** Each has its own `Address` object!

```
Original Person → name: "Alice"
                  age: 30
                  address: [Address Object 1] - "Boston"

Cloned Person →   name: "Bob"
                  age: 25
                  address: [Address Object 2] - "New York"
```

---

## Basic Implementation

### Method 1: Using Python's `copy` Module

```python
import copy
from abc import ABC, abstractmethod

class Prototype(ABC):
    """Abstract base class for prototypes"""
    
    @abstractmethod
    def clone(self):
        """Clone the object"""
        pass

class ConcretePrototype(Prototype):
    """Concrete prototype implementation"""
    
    def __init__(self, name, value, items=None):
        self.name = name
        self.value = value
        self.items = items if items is not None else []
    
    def clone(self):
        """Create a deep copy of this object"""
        return copy.deepcopy(self)
    
    def __str__(self):
        return f"ConcretePrototype(name={self.name}, value={self.value}, items={self.items})"

# Usage
original = ConcretePrototype("Original", 100, ["item1", "item2"])
print("Original:", original)

# Clone
clone1 = original.clone()
clone1.name = "Clone 1"
clone1.value = 200
clone1.items.append("item3")  # Modify clone's list

print("\nAfter modifying clone:")
print("Original:", original)  # Unchanged
print("Clone 1:", clone1)     # Modified

# Another clone
clone2 = original.clone()
clone2.name = "Clone 2"
print("Clone 2:", clone2)
```

**Output:**
```
Original: ConcretePrototype(name=Original, value=100, items=['item1', 'item2'])

After modifying clone:
Original: ConcretePrototype(name=Original, value=100, items=['item1', 'item2'])
Clone 1: ConcretePrototype(name=Clone 1, value=200, items=['item1', 'item2', 'item3'])
Clone 2: ConcretePrototype(name=Clone 2, value=100, items=['item1', 'item2'])
```

---

### Method 2: Custom Clone Implementation

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract Shape class"""
    
    def __init__(self, color, x, y):
        self.color = color
        self.x = x
        self.y = y
    
    @abstractmethod
    def clone(self):
        """Clone this shape"""
        pass
    
    @abstractmethod
    def draw(self):
        """Draw the shape"""
        pass

class Circle(Shape):
    """Circle implementation"""
    
    def __init__(self, color, x, y, radius):
        super().__init__(color, x, y)
        self.radius = radius
    
    def clone(self):
        """Create a copy of this circle"""
        return Circle(self.color, self.x, self.y, self.radius)
    
    def draw(self):
        return f"Circle(color={self.color}, pos=({self.x},{self.y}), radius={self.radius})"

class Rectangle(Shape):
    """Rectangle implementation"""
    
    def __init__(self, color, x, y, width, height):
        super().__init__(color, x, y)
        self.width = width
        self.height = height
    
    def clone(self):
        """Create a copy of this rectangle"""
        return Rectangle(self.color, self.x, self.y, self.width, self.height)
    
    def draw(self):
        return f"Rectangle(color={self.color}, pos=({self.x},{self.y}), size={self.width}x{self.height})"

# Usage
circle1 = Circle("red", 10, 20, 5)
print(circle1.draw())

# Clone and modify
circle2 = circle1.clone()
circle2.x = 100
circle2.color = "blue"

print(circle1.draw())  # Original unchanged
print(circle2.draw())  # Clone modified

# Rectangle
rect1 = Rectangle("green", 0, 0, 50, 30)
rect2 = rect1.clone()
rect2.width = 100

print(rect1.draw())  # Original unchanged
print(rect2.draw())  # Clone modified
```

---

## Prototype Registry

A registry manages a collection of prototypes and provides access to them by name.

```python
import copy
from typing import Dict

class Prototype:
    """Base prototype class"""
    
    def clone(self):
        return copy.deepcopy(self)

class PrototypeRegistry:
    """
    Registry to manage and access prototypes.
    Implements the Prototype Pattern with a registry.
    """
    
    def __init__(self):
        self._prototypes: Dict[str, Prototype] = {}
    
    def register(self, name: str, prototype: Prototype):
        """Register a prototype with a name"""
        self._prototypes[name] = prototype
        print(f"Registered prototype: {name}")
    
    def unregister(self, name: str):
        """Remove a prototype from registry"""
        if name in self._prototypes:
            del self._prototypes[name]
            print(f"Unregistered prototype: {name}")
    
    def clone(self, name: str) -> Prototype:
        """Clone a prototype by name"""
        prototype = self._prototypes.get(name)
        if prototype is None:
            raise ValueError(f"Prototype '{name}' not found in registry")
        return prototype.clone()
    
    def list_prototypes(self):
        """List all registered prototypes"""
        return list(self._prototypes.keys())

# Concrete prototypes
class Circle(Prototype):
    def __init__(self, radius, color):
        self.radius = radius
        self.color = color
    
    def __str__(self):
        return f"Circle(radius={self.radius}, color={self.color})"

class Rectangle(Prototype):
    def __init__(self, width, height, color):
        self.width = width
        self.height = height
        self.color = color
    
    def __str__(self):
        return f"Rectangle(width={self.width}, height={self.height}, color={self.color})"

# Usage
registry = PrototypeRegistry()

# Create and register prototypes
small_red_circle = Circle(5, "red")
large_blue_circle = Circle(20, "blue")
default_rectangle = Rectangle(10, 15, "green")

registry.register("small-circle", small_red_circle)
registry.register("large-circle", large_blue_circle)
registry.register("default-rect", default_rectangle)

print("\nAvailable prototypes:", registry.list_prototypes())

# Clone from registry
circle1 = registry.clone("small-circle")
print("\nCloned:", circle1)

circle2 = registry.clone("large-circle")
circle2.color = "yellow"  # Modify clone
print("Modified clone:", circle2)

rect1 = registry.clone("default-rect")
print("Cloned rectangle:", rect1)

# Original prototypes unchanged
print("\nOriginal small-circle:", small_red_circle)
print("Original large-circle:", large_blue_circle)
```

**Output:**
```
Registered prototype: small-circle
Registered prototype: large-circle
Registered prototype: default-rect

Available prototypes: ['small-circle', 'large-circle', 'default-rect']

Cloned: Circle(radius=5, color=red)
Modified clone: Circle(radius=20, color=yellow)
Cloned rectangle: Rectangle(width=10, height=15, color=green)

Original small-circle: Circle(radius=5, color=red)
Original large-circle: Circle(radius=20, color=blue)
```

---

## Real-World Examples

### Example 1: Game Development - Enemy Spawner

```python
import copy
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum

class EnemyType(Enum):
    GOBLIN = "goblin"
    ORC = "orc"
    DRAGON = "dragon"
    SKELETON = "skeleton"

@dataclass
class Position:
    x: float
    y: float
    
    def __str__(self):
        return f"({self.x}, {self.y})"

@dataclass
class Stats:
    health: int
    damage: int
    speed: float
    defense: int

class Enemy:
    """Enemy prototype"""
    
    def __init__(self, enemy_type: EnemyType, stats: Stats, 
                 inventory: List[str] = None, abilities: List[str] = None):
        self.enemy_type = enemy_type
        self.stats = stats
        self.inventory = inventory if inventory else []
        self.abilities = abilities if abilities else []
        self.position = Position(0, 0)
        self.is_alive = True
    
    def clone(self):
        """Create a deep copy of this enemy"""
        return copy.deepcopy(self)
    
    def spawn_at(self, x: float, y: float):
        """Set spawn position"""
        self.position = Position(x, y)
        return self
    
    def __str__(self):
        return (f"{self.enemy_type.value.capitalize()} at {self.position} - "
                f"HP: {self.stats.health}, DMG: {self.stats.damage}, "
                f"Inventory: {self.inventory}")

class EnemyPrototypeRegistry:
    """Registry for enemy prototypes"""
    
    def __init__(self):
        self._prototypes: Dict[EnemyType, Enemy] = {}
        self._initialize_prototypes()
    
    def _initialize_prototypes(self):
        """Create standard enemy prototypes"""
        
        # Goblin - weak but fast
        goblin = Enemy(
            EnemyType.GOBLIN,
            Stats(health=50, damage=10, speed=2.0, defense=5),
            inventory=["rusty_dagger", "gold_coin"],
            abilities=["quick_attack"]
        )
        self.register(EnemyType.GOBLIN, goblin)
        
        # Orc - strong and slow
        orc = Enemy(
            EnemyType.ORC,
            Stats(health=150, damage=30, speed=0.8, defense=15),
            inventory=["battle_axe", "leather_armor", "health_potion"],
            abilities=["power_strike", "intimidate"]
        )
        self.register(EnemyType.ORC, orc)
        
        # Dragon - very powerful
        dragon = Enemy(
            EnemyType.DRAGON,
            Stats(health=500, damage=100, speed=1.5, defense=50),
            inventory=["dragon_scales", "treasure_hoard"],
            abilities=["fire_breath", "fly", "tail_swipe"]
        )
        self.register(EnemyType.DRAGON, dragon)
        
        # Skeleton - undead
        skeleton = Enemy(
            EnemyType.SKELETON,
            Stats(health=30, damage=15, speed=1.2, defense=8),
            inventory=["bone", "old_sword"],
            abilities=["undead_resilience"]
        )
        self.register(EnemyType.SKELETON, skeleton)
    
    def register(self, enemy_type: EnemyType, prototype: Enemy):
        """Register a prototype"""
        self._prototypes[enemy_type] = prototype
    
    def spawn(self, enemy_type: EnemyType, x: float = 0, y: float = 0) -> Enemy:
        """Spawn a new enemy from prototype"""
        if enemy_type not in self._prototypes:
            raise ValueError(f"No prototype for {enemy_type}")
        
        enemy = self._prototypes[enemy_type].clone()
        enemy.spawn_at(x, y)
        return enemy
    
    def spawn_wave(self, wave_composition: Dict[EnemyType, int], 
                   start_x: float = 0, start_y: float = 0) -> List[Enemy]:
        """Spawn multiple enemies"""
        enemies = []
        offset = 0
        
        for enemy_type, count in wave_composition.items():
            for i in range(count):
                enemy = self.spawn(enemy_type, start_x + offset, start_y)
                enemies.append(enemy)
                offset += 10  # Space them out
        
        return enemies

# ============ USAGE ============

# Create enemy spawner
spawner = EnemyPrototypeRegistry()

print("=== Spawning Individual Enemies ===")
goblin1 = spawner.spawn(EnemyType.GOBLIN, 100, 50)
goblin2 = spawner.spawn(EnemyType.GOBLIN, 150, 50)
orc1 = spawner.spawn(EnemyType.ORC, 200, 100)

print(goblin1)
print(goblin2)
print(orc1)

print("\n=== Spawning Enemy Wave ===")
wave1_composition = {
    EnemyType.GOBLIN: 5,
    EnemyType.SKELETON: 3,
    EnemyType.ORC: 2
}

wave1 = spawner.spawn_wave(wave1_composition, start_x=0, start_y=0)
print(f"Spawned {len(wave1)} enemies in wave 1:")
for enemy in wave1:
    print(f"  - {enemy}")

print("\n=== Boss Wave ===")
wave2_composition = {
    EnemyType.DRAGON: 1,
    EnemyType.ORC: 4
}

wave2 = spawner.spawn_wave(wave2_composition, start_x=500, start_y=500)
print(f"Spawned {len(wave2)} enemies in boss wave:")
for enemy in wave2:
    print(f"  - {enemy}")

# Modify a clone without affecting prototype
print("\n=== Modifying Clone ===")
special_goblin = spawner.spawn(EnemyType.GOBLIN, 300, 300)
special_goblin.stats.health = 100  # Power up this goblin
special_goblin.inventory.append("magic_sword")
print("Special goblin:", special_goblin)

# Spawn normal goblin - unchanged
normal_goblin = spawner.spawn(EnemyType.GOBLIN, 350, 300)
print("Normal goblin:", normal_goblin)
```

---

### Example 2: Document Templates

```python
import copy
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

class DocumentType(Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    LETTER = "letter"

class Section:
    """Document section"""
    
    def __init__(self, title: str, content: str = ""):
        self.title = title
        self.content = content
    
    def __str__(self):
        return f"\n{'='*50}\n{self.title}\n{'='*50}\n{self.content}\n"

class Document:
    """Document prototype"""
    
    def __init__(self, doc_type: DocumentType, title: str):
        self.doc_type = doc_type
        self.title = title
        self.created_at = datetime.now()
        self.sections: List[Section] = []
        self.metadata: Dict[str, str] = {}
        self.header = ""
        self.footer = ""
    
    def add_section(self, section: Section):
        """Add a section to document"""
        self.sections.append(section)
        return self
    
    def set_header(self, header: str):
        """Set document header"""
        self.header = header
        return self
    
    def set_footer(self, footer: str):
        """Set document footer"""
        self.footer = footer
        return self
    
    def set_metadata(self, key: str, value: str):
        """Set metadata"""
        self.metadata[key] = value
        return self
    
    def clone(self):
        """Clone this document"""
        cloned = copy.deepcopy(self)
        cloned.created_at = datetime.now()  # Update timestamp
        return cloned
    
    def render(self) -> str:
        """Render the document"""
        output = []
        
        # Header
        if self.header:
            output.append(self.header)
            output.append("\n")
        
        # Title
        output.append(f"\n{'#'*60}")
        output.append(f"\n{self.title.center(60)}")
        output.append(f"\n{'#'*60}\n")
        
        # Metadata
        if self.metadata:
            output.append("\nDocument Information:")
            for key, value in self.metadata.items():
                output.append(f"  {key}: {value}")
            output.append("\n")
        
        # Created date
        output.append(f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Sections
        for section in self.sections:
            output.append(str(section))
        
        # Footer
        if self.footer:
            output.append("\n")
            output.append(self.footer)
        
        return "".join(output)

class DocumentTemplateRegistry:
    """Registry for document templates"""
    
    def __init__(self):
        self._templates: Dict[DocumentType, Document] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Create standard document templates"""
        
        # Invoice template
        invoice = Document(DocumentType.INVOICE, "INVOICE")
        invoice.set_header("ACME Corporation\n123 Business St, City, State 12345\nPhone: (555) 123-4567")
        invoice.set_footer("Thank you for your business!\nPayment due within 30 days.")
        invoice.add_section(Section("Bill To", "[Customer Name]\n[Address]"))
        invoice.add_section(Section("Items", "[Item list will go here]"))
        invoice.add_section(Section("Total", "[Total amount]"))
        invoice.set_metadata("Type", "Invoice")
        invoice.set_metadata("Terms", "Net 30")
        self.register(DocumentType.INVOICE, invoice)
        
        # Receipt template
        receipt = Document(DocumentType.RECEIPT, "RECEIPT")
        receipt.set_header("ACME Corporation\nThank you for your purchase!")
        receipt.set_footer("Please retain this receipt for your records.")
        receipt.add_section(Section("Transaction Details", "[Details here]"))
        receipt.add_section(Section("Payment Method", "[Payment info]"))
        receipt.add_section(Section("Amount Paid", "[Amount]"))
        self.register(DocumentType.RECEIPT, receipt)
        
        # Contract template
        contract = Document(DocumentType.CONTRACT, "SERVICE CONTRACT")
        contract.set_header("PROFESSIONAL SERVICES AGREEMENT")
        contract.set_footer("Signatures:\n\n__________________    __________________\nClient                Service Provider")
        contract.add_section(Section("Parties", "This agreement is between [Party A] and [Party B]"))
        contract.add_section(Section("Scope of Work", "[Scope details]"))
        contract.add_section(Section("Terms and Conditions", "[Terms]"))
        contract.add_section(Section("Payment Terms", "[Payment details]"))
        self.register(DocumentType.CONTRACT, contract)
        
        # Letter template
        letter = Document(DocumentType.LETTER, "BUSINESS LETTER")
        letter.set_header("ACME Corporation")
        letter.add_section(Section("Recipient", "[Recipient Name]\n[Address]"))
        letter.add_section(Section("Body", "Dear [Name],\n\n[Letter content]\n\nSincerely,\n[Your Name]"))
        self.register(DocumentType.LETTER, letter)
    
    def register(self, doc_type: DocumentType, template: Document):
        """Register a template"""
        self._templates[doc_type] = template
    
    def create(self, doc_type: DocumentType) -> Document:
        """Create a new document from template"""
        if doc_type not in self._templates:
            raise ValueError(f"No template for {doc_type}")
        return self._templates[doc_type].clone()

# ============ USAGE ============

# Create template registry
templates = DocumentTemplateRegistry()

print("=== Creating Invoice from Template ===")
invoice1 = templates.create(DocumentType.INVOICE)
invoice1.title = "INVOICE #001"
invoice1.set_metadata("Invoice Number", "INV-001")
invoice1.set_metadata("Customer", "John Doe")
invoice1.sections[0].content = "John Doe\n456 Customer Ave\nCity, State 67890"
invoice1.sections[1].content = "1x Widget @ $50.00\n2x Gadget @ $30.00\nSubtotal: $110.00"
invoice1.sections[2].content = "Total: $110.00"

print(invoice1.render())

print("\n" + "="*80 + "\n")

print("=== Creating Another Invoice ===")
invoice2 = templates.create(DocumentType.INVOICE)
invoice2.title = "INVOICE #002"
invoice2.set_metadata("Invoice Number", "INV-002")
invoice2.set_metadata("Customer", "Jane Smith")
invoice2.sections[0].content = "Jane Smith\n789 Client Rd\nCity, State 11111"
invoice2.sections[1].content = "5x Service Hours @ $100.00\nSubtotal: $500.00"
invoice2.sections[2].content = "Total: $500.00"

print(invoice2.render())

print("\n" + "="*80 + "\n")

print("=== Creating Receipt from Template ===")
receipt = templates.create(DocumentType.RECEIPT)
receipt.title = "RECEIPT #R-001"
receipt.sections[0].content = "Date: 2024-02-08\nTransaction ID: TXN-12345"
receipt.sections[1].content = "Credit Card ending in 4242"
receipt.sections[2].content = "$110.00"

print(receipt.render())
```

---

### Example 3: UI Component Library

```python
import copy
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class Style:
    """Styling information"""
    background_color: str = "white"
    text_color: str = "black"
    border_color: str = "gray"
    border_width: int = 1
    padding: int = 10
    font_size: int = 14
    font_weight: str = "normal"

class UIComponent:
    """Base UI component"""
    
    def __init__(self, component_type: str, text: str = ""):
        self.component_type = component_type
        self.text = text
        self.style = Style()
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.enabled = True
        self.visible = True
        self.children: List['UIComponent'] = []
    
    def set_style(self, **kwargs):
        """Update style properties"""
        for key, value in kwargs.items():
            if hasattr(self.style, key):
                setattr(self.style, key, value)
        return self
    
    def add_child(self, child: 'UIComponent'):
        """Add child component"""
        self.children.append(child)
        return self
    
    def clone(self):
        """Clone this component"""
        return copy.deepcopy(self)
    
    def render(self, indent=0) -> str:
        """Render component as text"""
        prefix = "  " * indent
        output = [f"{prefix}<{self.component_type}"]
        
        if self.text:
            output.append(f' text="{self.text}"')
        if self.width:
            output.append(f' width={self.width}')
        if self.height:
            output.append(f' height={self.height}')
        
        output.append(f' style="bg:{self.style.background_color}, color:{self.style.text_color}"')
        output.append(">")
        
        result = "".join(output)
        
        if self.children:
            result += "\n"
            for child in self.children:
                result += child.render(indent + 1) + "\n"
            result += f"{prefix}</{self.component_type}>"
        else:
            result += f"</{self.component_type}>"
        
        return result

class UIComponentLibrary:
    """Library of pre-configured UI components"""
    
    def __init__(self):
        self._prototypes: Dict[str, UIComponent] = {}
        self._initialize_library()
    
    def _initialize_library(self):
        """Create standard UI component prototypes"""
        
        # Primary Button
        primary_btn = UIComponent("Button", "Click Me")
        primary_btn.set_style(
            background_color="#007bff",
            text_color="white",
            border_color="#0056b3",
            padding=12,
            font_weight="bold"
        )
        primary_btn.width = 120
        primary_btn.height = 40
        self.register("primary-button", primary_btn)
        
        # Secondary Button
        secondary_btn = UIComponent("Button", "Cancel")
        secondary_btn.set_style(
            background_color="#6c757d",
            text_color="white",
            border_color="#545b62",
            padding=12
        )
        secondary_btn.width = 120
        secondary_btn.height = 40
        self.register("secondary-button", secondary_btn)
        
        # Danger Button
        danger_btn = UIComponent("Button", "Delete")
        danger_btn.set_style(
            background_color="#dc3545",
            text_color="white",
            border_color="#bd2130",
            padding=12,
            font_weight="bold"
        )
        danger_btn.width = 120
        danger_btn.height = 40
        self.register("danger-button", danger_btn)
        
        # Text Input
        text_input = UIComponent("Input", "")
        text_input.set_style(
            background_color="white",
            text_color="black",
            border_color="#ced4da",
            border_width=1,
            padding=8
        )
        text_input.width = 200
        text_input.height = 35
        self.register("text-input", text_input)
        
        # Card
        card = UIComponent("Card")
        card.set_style(
            background_color="white",
            border_color="#dee2e6",
            border_width=1,
            padding=20
        )
        card.width = 300
        self.register("card", card)
        
        # Alert - Success
        success_alert = UIComponent("Alert", "Operation successful!")
        success_alert.set_style(
            background_color="#d4edda",
            text_color="#155724",
            border_color="#c3e6cb",
            padding=15
        )
        self.register("success-alert", success_alert)
        
        # Alert - Error
        error_alert = UIComponent("Alert", "An error occurred!")
        error_alert.set_style(
            background_color="#f8d7da",
            text_color="#721c24",
            border_color="#f5c6cb",
            padding=15
        )
        self.register("error-alert", error_alert)
        
        # Form
        form = UIComponent("Form")
        form.set_style(padding=20, background_color="#f8f9fa")
        form.width = 400
        self.register("standard-form", form)
    
    def register(self, name: str, prototype: UIComponent):
        """Register a component prototype"""
        self._prototypes[name] = prototype
    
    def create(self, name: str) -> UIComponent:
        """Create component from prototype"""
        if name not in self._prototypes:
            raise ValueError(f"No prototype named '{name}'")
        return self._prototypes[name].clone()
    
    def list_components(self):
        """List all available components"""
        return list(self._prototypes.keys())

# ============ USAGE ============

# Create component library
library = UIComponentLibrary()

print("Available components:", library.list_components())
print("\n" + "="*80 + "\n")

# Create a login form using prototypes
print("=== Building Login Form ===\n")

form = library.create("standard-form")
form.text = "Login Form"

# Add components to form
username_input = library.create("text-input")
username_input.text = "Username"

password_input = library.create("text-input")
password_input.text = "Password"

login_button = library.create("primary-button")
login_button.text = "Login"

cancel_button = library.create("secondary-button")
cancel_button.text = "Cancel"

form.add_child(username_input)
form.add_child(password_input)
form.add_child(login_button)
form.add_child(cancel_button)

print(form.render())

print("\n" + "="*80 + "\n")

# Create a card with alert
print("=== Building Card with Alert ===\n")

card = library.create("card")
success_alert = library.create("success-alert")
success_alert.text = "Your profile has been updated!"

primary_btn = library.create("primary-button")
primary_btn.text = "OK"

card.add_child(success_alert)
card.add_child(primary_btn)

print(card.render())

print("\n" + "="*80 + "\n")

# Create multiple buttons
print("=== Creating Button Group ===\n")

save_btn = library.create("primary-button")
save_btn.text = "Save"

edit_btn = library.create("secondary-button")
edit_btn.text = "Edit"

delete_btn = library.create("danger-button")
delete_btn.text = "Delete"

print(save_btn.render())
print(edit_btn.render())
print(delete_btn.render())
```

---

## Common Pitfalls

### ❌ Pitfall 1: Forgetting Deep Copy for Nested Objects

```python
import copy

class BadClone:
    def __init__(self):
        self.items = []
    
    def clone(self):
        new_obj = BadClone()
        new_obj.items = self.items  # ⚠️ Shallow copy - shares list!
        return new_obj

# Problem
original = BadClone()
original.items.append("item1")

clone = original.clone()
clone.items.append("item2")  # Affects original too!

print(original.items)  # ['item1', 'item2'] - Oops!

# GOOD - Deep copy
class GoodClone:
    def __init__(self):
        self.items = []
    
    def clone(self):
        return copy.deepcopy(self)  # ✅ Deep copy

original2 = GoodClone()
original2.items.append("item1")

clone2 = original2.clone()
clone2.items.append("item2")

print(original2.items)  # ['item1'] - Correct!
print(clone2.items)     # ['item1', 'item2']
```

---

### ❌ Pitfall 2: Circular References

```python
import copy

class Node:
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = []
    
    def add_child(self, child):
        child.parent = self  # Circular reference!
        self.children.append(child)
    
    def clone(self):
        # This works but be careful with circular refs
        return copy.deepcopy(self)

# Usage
root = Node("root")
child1 = Node("child1")
root.add_child(child1)

# Deep copy handles circular refs correctly
cloned_root = root.clone()
print(cloned_root.children[0].parent.value)  # "root" - works!
```

---

### ❌ Pitfall 3: Cloning Singletons

```python
# BAD - Don't clone singletons!
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def clone(self):
        return copy.deepcopy(self)  # ⚠️ Breaks singleton!

# Problem
s1 = Singleton()
s2 = s1.clone()  # Now you have 2 instances!
print(s1 is s2)  # False - broke singleton pattern
```

---

### ❌ Pitfall 4: Not Resetting Mutable Defaults

```python
# BAD
class BadPrototype:
    def __init__(self, items=[]):  # ⚠️ Mutable default
        self.items = items

# GOOD
class GoodPrototype:
    def __init__(self, items=None):
        self.items = items if items is not None else []
```

---

## Best Practices

### ✅ 1. Use `copy.deepcopy()` for Complex Objects

```python
import copy

class Component:
    def clone(self):
        return copy.deepcopy(self)  # Safe and easy
```

---

### ✅ 2. Implement Custom Clone for Performance

```python
# If deep copy is too slow, implement custom clone
class OptimizedPrototype:
    def __init__(self, heavy_data, metadata):
        self.heavy_data = heavy_data  # Large, immutable
        self.metadata = metadata      # Small, mutable
    
    def clone(self):
        # Share immutable data, copy mutable data
        new_obj = OptimizedPrototype(
            heavy_data=self.heavy_data,  # Share (immutable)
            metadata=self.metadata.copy()  # Copy (mutable)
        )
        return new_obj
```

---

### ✅ 3. Document Cloning Behavior

```python
class Document:
    """
    Document prototype.
    
    Cloning behavior:
    - Deep copies all sections and metadata
    - Resets created_at timestamp to current time
    - Preserves template structure
    """
    
    def clone(self):
        cloned = copy.deepcopy(self)
        cloned.created_at = datetime.now()
        return cloned
```

---

### ✅ 4. Use Registry for Common Prototypes

```python
class PrototypeManager:
    """Centralized prototype management"""
    
    def __init__(self):
        self._prototypes = {}
    
    def register(self, name, prototype):
        self._prototypes[name] = prototype
    
    def clone(self, name):
        return self._prototypes[name].clone()
```

---

### ✅ 5. Consider Using `__copy__` and `__deepcopy__`

```python
import copy

class CustomClone:
    def __init__(self, value, items):
        self.value = value
        self.items = items
    
    def __copy__(self):
        """Shallow copy"""
        return CustomClone(self.value, self.items)
    
    def __deepcopy__(self, memo):
        """Deep copy"""
        return CustomClone(
            copy.deepcopy(self.value, memo),
            copy.deepcopy(self.items, memo)
        )

# Now works with copy module
obj = CustomClone(10, [1, 2, 3])
shallow = copy.copy(obj)
deep = copy.deepcopy(obj)
```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Purpose** | Clone existing objects instead of creating new ones |
| **Use When** | Object creation is expensive, need many similar objects |
| **Avoid When** | Simple objects, cloning more expensive than creating |
| **Key Consideration** | Shallow vs Deep copy - understand the difference! |
| **Common Use Cases** | Game development, templates, UI components |

---

**Comparison Table:**

| Pattern | Creation Method | Use Case | Flexibility |
|---------|----------------|----------|-------------|
| **Factory** | Calls constructor | Different types needed | Medium |
| **Builder** | Step-by-step construction | Complex objects | High |
| **Prototype** | Clones existing object | Many similar objects | Medium |
| **Singleton** | Controlled instantiation | One instance only | Low |
