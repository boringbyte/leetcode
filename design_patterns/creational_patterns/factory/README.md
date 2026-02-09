# Factory Pattern - Complete Guide

## 📋 Table of Contents
- [What is Factory Pattern?](#what-is-factory-pattern)
- [Types of Factory Patterns](#types-of-factory-patterns)
- [When to Use](#when-to-use)
- [When NOT to Use](#when-not-to-use)
- [Simple Factory](#simple-factory)
- [Factory Method Pattern](#factory-method-pattern)
- [Abstract Factory Pattern](#abstract-factory-pattern)
- [Real-World Examples](#real-world-examples)
- [Common Pitfalls](#common-pitfalls)
- [Best Practices](#best-practices)

---

## What is Factory Pattern?

**Factory Pattern** is a creational design pattern that provides an interface for creating objects without specifying their exact classes. It delegates the instantiation logic to specialized factory classes or methods.

### Key Characteristics:
- ✅ Encapsulates object creation logic
- ✅ Client code doesn't need to know concrete class names
- ✅ Makes code more flexible and easier to extend
- ✅ Follows Open/Closed Principle (open for extension, closed for modification)

### Visual Representation:
```
Client → Factory.create("type") → Returns appropriate object
         ↓
    Creates either: ObjectA, ObjectB, or ObjectC
    (Client doesn't know which one!)
```

---

## Types of Factory Patterns

There are **three main types**:

| Type                  | Complexity  | Use Case                                 | Creation logic lives in  | Python feature used | Best for                    |
|-----------------------|-------------|------------------------------------------|--------------------------|---------------------|-----------------------------|
| **Simple Factory**    | Low         | Single factory creates different objects | Function / single class  | if / match          | Small systems               |
| **Factory Method**    | Medium      | Subclasses decide which object to create | Subclasses               | Polymorphism        | Extensible systems          |
| **Abstract Factory**  | High        | Creates families of related objects      | Concrete factory classes | ABCs + composition  | Large, configurable systems |

---

## When to Use

### ✅ Perfect Use Cases:

#### 1. **Object Creation is Complex**
- Construction involves multiple steps
- Need to perform initialization logic
- Dependencies need to be set up

#### 2. **Don't Know Exact Type Until Runtime**
- Type depends on user input
- Type depends on configuration
- Type depends on data from API/database

#### 3. **Want to Decouple Client from Concrete Classes**
- Client shouldn't know implementation details
- Makes testing easier (can swap implementations)
- Reduces coupling

#### 4. **Need to Support Multiple Variants**
- Different payment methods (credit card, PayPal, crypto)
- Different file formats (PDF, Word, Excel)
- Different notification channels (Email, SMS, Push)

#### 5. **Want to Follow Open/Closed Principle**
- Add new types without modifying existing code
- Extend functionality without breaking existing code

---

## When NOT to Use

### ❌ Avoid Factory When:

1. **Only One Type Exists**
   - No variation in objects being created
   - Adds unnecessary complexity

2. **Object Creation is Trivial**
   - Simple `__init__` with few parameters
   - No complex initialization needed

3. **You Need Direct Control**
   - Need to pass specific parameters to constructor
   - Factory pattern can hide important details

4. **Over-Engineering Simple Code**
   - Don't use it "just in case" you might need it later
   - YAGNI (You Aren't Gonna Need It)

---

## Simple Factory

**Concept:** A single factory class with a method that returns different object types based on input.

### Basic Implementation

```python
from abc import ABC, abstractmethod

# Product Interface
class Animal(ABC):
    @abstractmethod
    def speak(self):
        pass
    
    @abstractmethod
    def move(self):
        pass

# Concrete Products
class Dog(Animal):
    def speak(self):
        return "Woof! 🐕"
    
    def move(self):
        return "Running on four legs"

class Cat(Animal):
    def speak(self):
        return "Meow! 🐈"
    
    def move(self):
        return "Sneaking silently"

class Bird(Animal):
    def speak(self):
        return "Tweet! 🐦"
    
    def move(self):
        return "Flying in the sky"

# Simple Factory
class AnimalFactory:
    @staticmethod
    def create_animal(animal_type: str) -> Animal:
        """
        Factory method to create animals
        
        Args:
            animal_type: Type of animal ("dog", "cat", "bird")
            
        Returns:
            Animal instance
            
        Raises:
            ValueError: If animal_type is unknown
        """
        animal_type = animal_type.lower()
        
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        elif animal_type == "bird":
            return Bird()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

# Usage
factory = AnimalFactory()

# Client doesn't need to know about Dog, Cat, Bird classes
animal1 = factory.create_animal("dog")
print(animal1.speak())  # Woof! 🐕
print(animal1.move())   # Running on four legs

animal2 = factory.create_animal("cat")
print(animal2.speak())  # Meow! 🐈

animal3 = factory.create_animal("bird")
print(animal3.speak())  # Tweet! 🐦
```

### Better: Using Dictionary Mapping

```python
class AnimalFactory:
    # Registry of available animals
    _animal_types = {
        "dog": Dog,
        "cat": Cat,
        "bird": Bird
    }
    
    @classmethod
    def create_animal(cls, animal_type: str) -> Animal:
        animal_type = animal_type.lower()
        
        animal_class = cls._animal_types.get(animal_type)
        if animal_class is None:
            raise ValueError(f"Unknown animal type: {animal_type}")
        
        return animal_class()
    
    @classmethod
    def register_animal(cls, name: str, animal_class):
        """Allow adding new animal types dynamically"""
        cls._animal_types[name] = animal_class
    
    @classmethod
    def get_available_types(cls):
        """Get list of available animal types"""
        return list(cls._animal_types.keys())

# Usage
factory = AnimalFactory()
animal = factory.create_animal("dog")

# Check available types
print(factory.get_available_types())  # ['dog', 'cat', 'bird']

# Add new type dynamically
class Fish(Animal):
    def speak(self):
        return "Blub! 🐟"
    
    def move(self):
        return "Swimming"

factory.register_animal("fish", Fish)
fish = factory.create_animal("fish")
print(fish.speak())  # Blub! 🐟
```

---

## Factory Method Pattern

**Concept:** Define an interface for creating objects, but let subclasses decide which class to instantiate.

### Structure

```python
from abc import ABC, abstractmethod

# Product Interface
class Document(ABC):
    @abstractmethod
    def open(self):
        pass
    
    @abstractmethod
    def save(self):
        pass
    
    @abstractmethod
    def close(self):
        pass

# Concrete Products
class PDFDocument(Document):
    def open(self):
        return "Opening PDF document..."
    
    def save(self):
        return "Saving as PDF..."
    
    def close(self):
        return "Closing PDF document"

class WordDocument(Document):
    def open(self):
        return "Opening Word document..."
    
    def save(self):
        return "Saving as DOCX..."
    
    def close(self):
        return "Closing Word document"

class ExcelDocument(Document):
    def open(self):
        return "Opening Excel spreadsheet..."
    
    def save(self):
        return "Saving as XLSX..."
    
    def close(self):
        return "Closing Excel spreadsheet"

# Creator (Abstract)
class Application(ABC):
    """
    The Creator class declares the factory method that returns
    new product objects. Subclasses implement this method.
    """
    
    @abstractmethod
    def create_document(self) -> Document:
        """Factory method - subclasses must implement"""
        pass
    
    def new_document(self):
        """
        Business logic that uses the factory method.
        Note: despite the name, the Creator's primary responsibility
        is not creating products. It usually contains core business logic
        that relies on product objects returned by the factory method.
        """
        doc = self.create_document()  # Factory method call
        print(doc.open())
        return doc
    
    def edit_document(self):
        doc = self.create_document()
        print("Editing document...")
        print(doc.save())

# Concrete Creators
class PDFApplication(Application):
    def create_document(self) -> Document:
        return PDFDocument()

class WordApplication(Application):
    def create_document(self) -> Document:
        return WordDocument()

class ExcelApplication(Application):
    def create_document(self) -> Document:
        return ExcelDocument()

# Usage
def client_code(app: Application):
    """
    Client code works with any Application subclass
    without knowing the concrete document type
    """
    doc = app.new_document()
    print(doc.save())
    print(doc.close())

# Different applications create different documents
print("=== PDF Application ===")
pdf_app = PDFApplication()
client_code(pdf_app)

print("\n=== Word Application ===")
word_app = WordApplication()
client_code(word_app)

print("\n=== Excel Application ===")
excel_app = ExcelApplication()
client_code(excel_app)
```

**Output:**
```
=== PDF Application ===
Opening PDF document...
Saving as PDF...
Closing PDF document

=== Word Application ===
Opening Word document...
Saving as DOCX...
Closing Word document

=== Excel Application ===
Opening Excel spreadsheet...
Saving as XLSX...
Closing Excel spreadsheet
```

---

## Abstract Factory Pattern

**Concept:** Provide an interface for creating families of related or dependent objects without specifying their concrete classes.

### Structure

```python
from abc import ABC, abstractmethod

# ============ ABSTRACT PRODUCTS ============

class Button(ABC):
    @abstractmethod
    def render(self):
        pass
    
    @abstractmethod
    def on_click(self):
        pass

class Checkbox(ABC):
    @abstractmethod
    def render(self):
        pass
    
    @abstractmethod
    def on_check(self):
        pass

class TextField(ABC):
    @abstractmethod
    def render(self):
        pass
    
    @abstractmethod
    def on_input(self):
        pass

# ============ CONCRETE PRODUCTS - Windows Family ============

class WindowsButton(Button):
    def render(self):
        return "Rendering Windows-style button with blue theme"
    
    def on_click(self):
        return "Windows button clicked - Playing system sound"

class WindowsCheckbox(Checkbox):
    def render(self):
        return "Rendering Windows-style checkbox"
    
    def on_check(self):
        return "Windows checkbox checked"

class WindowsTextField(TextField):
    def render(self):
        return "Rendering Windows-style text field"
    
    def on_input(self):
        return "Windows text field input received"

# ============ CONCRETE PRODUCTS - Mac Family ============

class MacButton(Button):
    def render(self):
        return "Rendering Mac-style button with rounded corners"
    
    def on_click(self):
        return "Mac button clicked - Subtle animation"

class MacCheckbox(Checkbox):
    def render(self):
        return "Rendering Mac-style checkbox with smooth toggle"
    
    def on_check(self):
        return "Mac checkbox checked with animation"

class MacTextField(TextField):
    def render(self):
        return "Rendering Mac-style text field with glow effect"
    
    def on_input(self):
        return "Mac text field input received"

# ============ CONCRETE PRODUCTS - Linux Family ============

class LinuxButton(Button):
    def render(self):
        return "Rendering Linux-style button (GTK theme)"
    
    def on_click(self):
        return "Linux button clicked"

class LinuxCheckbox(Checkbox):
    def render(self):
        return "Rendering Linux-style checkbox"
    
    def on_check(self):
        return "Linux checkbox checked"

class LinuxTextField(TextField):
    def render(self):
        return "Rendering Linux-style text field"
    
    def on_input(self):
        return "Linux text field input received"

# ============ ABSTRACT FACTORY ============

class GUIFactory(ABC):
    """
    Abstract Factory interface declares creation methods
    for each distinct product type
    """
    
    @abstractmethod
    def create_button(self) -> Button:
        pass
    
    @abstractmethod
    def create_checkbox(self) -> Checkbox:
        pass
    
    @abstractmethod
    def create_text_field(self) -> TextField:
        pass

# ============ CONCRETE FACTORIES ============

class WindowsFactory(GUIFactory):
    def create_button(self) -> Button:
        return WindowsButton()
    
    def create_checkbox(self) -> Checkbox:
        return WindowsCheckbox()
    
    def create_text_field(self) -> TextField:
        return WindowsTextField()

class MacFactory(GUIFactory):
    def create_button(self) -> Button:
        return MacButton()
    
    def create_checkbox(self) -> Checkbox:
        return MacCheckbox()
    
    def create_text_field(self) -> TextField:
        return MacTextField()

class LinuxFactory(GUIFactory):
    def create_button(self) -> Button:
        return LinuxButton()
    
    def create_checkbox(self) -> Checkbox:
        return LinuxCheckbox()
    
    def create_text_field(self) -> TextField:
        return LinuxTextField()

# ============ CLIENT CODE ============

class Application:
    """
    Client code works with factories and products only through
    abstract types (GUIFactory, Button, Checkbox, TextField).
    This lets you pass any factory or product subclass without breaking it.
    """
    
    def __init__(self, factory: GUIFactory):
        self.factory = factory
    
    def create_login_form(self):
        # Create UI elements using the factory
        button = self.factory.create_button()
        checkbox = self.factory.create_checkbox()
        text_field = self.factory.create_text_field()
        
        print("=== Creating Login Form ===")
        print(text_field.render())
        print(checkbox.render())
        print(button.render())
        
        print("\n=== User Interaction ===")
        print(text_field.on_input())
        print(checkbox.on_check())
        print(button.on_click())

# Usage - Determine OS and create appropriate factory
import platform

def get_factory() -> GUIFactory:
    os_name = platform.system()
    
    if os_name == "Windows":
        return WindowsFactory()
    elif os_name == "Darwin":  # macOS
        return MacFactory()
    else:  # Linux
        return LinuxFactory()

# Application automatically uses the correct UI family
factory = get_factory()
app = Application(factory)
app.create_login_form()

# You can also explicitly choose
print("\n" + "="*50)
print("Forcing Mac UI on any platform:")
print("="*50)
mac_app = Application(MacFactory())
mac_app.create_login_form()
```

---

## Real-World Examples

### Example 1: Payment Processing System

```python
from abc import ABC, abstractmethod
from typing import Dict

# ============ PRODUCT INTERFACE ============

class PaymentProcessor(ABC):
    @abstractmethod
    def validate_credentials(self, credentials: Dict) -> bool:
        pass
    
    @abstractmethod
    def process_payment(self, amount: float, currency: str) -> Dict:
        pass
    
    @abstractmethod
    def refund_payment(self, transaction_id: str, amount: float) -> Dict:
        pass

# ============ CONCRETE PRODUCTS ============

class CreditCardProcessor(PaymentProcessor):
    def validate_credentials(self, credentials: Dict) -> bool:
        # Check card number, CVV, expiry
        required = ['card_number', 'cvv', 'expiry']
        return all(key in credentials for key in required)
    
    def process_payment(self, amount: float, currency: str) -> Dict:
        print(f"Processing ${amount} {currency} via Credit Card")
        # Actual credit card API call would go here
        return {
            'status': 'success',
            'transaction_id': 'CC-12345',
            'amount': amount,
            'method': 'credit_card'
        }
    
    def refund_payment(self, transaction_id: str, amount: float) -> Dict:
        print(f"Refunding ${amount} to credit card (Transaction: {transaction_id})")
        return {'status': 'refunded', 'transaction_id': transaction_id}

class PayPalProcessor(PaymentProcessor):
    def validate_credentials(self, credentials: Dict) -> bool:
        required = ['email', 'password']
        return all(key in credentials for key in required)
    
    def process_payment(self, amount: float, currency: str) -> Dict:
        print(f"Processing ${amount} {currency} via PayPal")
        return {
            'status': 'success',
            'transaction_id': 'PP-67890',
            'amount': amount,
            'method': 'paypal'
        }
    
    def refund_payment(self, transaction_id: str, amount: float) -> Dict:
        print(f"Refunding ${amount} via PayPal (Transaction: {transaction_id})")
        return {'status': 'refunded', 'transaction_id': transaction_id}

class CryptoProcessor(PaymentProcessor):
    def validate_credentials(self, credentials: Dict) -> bool:
        required = ['wallet_address', 'private_key']
        return all(key in credentials for key in required)
    
    def process_payment(self, amount: float, currency: str) -> Dict:
        print(f"Processing {amount} {currency} via Cryptocurrency")
        return {
            'status': 'pending',  # Crypto takes time
            'transaction_id': 'CRYPTO-ABCDE',
            'amount': amount,
            'method': 'crypto'
        }
    
    def refund_payment(self, transaction_id: str, amount: float) -> Dict:
        print(f"Initiating crypto refund of {amount} (Transaction: {transaction_id})")
        return {'status': 'refund_initiated', 'transaction_id': transaction_id}

class BankTransferProcessor(PaymentProcessor):
    def validate_credentials(self, credentials: Dict) -> bool:
        required = ['account_number', 'routing_number']
        return all(key in credentials for key in required)
    
    def process_payment(self, amount: float, currency: str) -> Dict:
        print(f"Processing ${amount} {currency} via Bank Transfer")
        return {
            'status': 'processing',  # Bank transfers take days
            'transaction_id': 'BANK-54321',
            'amount': amount,
            'method': 'bank_transfer'
        }
    
    def refund_payment(self, transaction_id: str, amount: float) -> Dict:
        print(f"Initiating bank refund of ${amount} (Transaction: {transaction_id})")
        return {'status': 'refund_processing', 'transaction_id': transaction_id}

# ============ FACTORY ============

class PaymentProcessorFactory:
    """Factory to create payment processors"""
    
    _processors = {
        'credit_card': CreditCardProcessor,
        'paypal': PayPalProcessor,
        'crypto': CryptoProcessor,
        'bank_transfer': BankTransferProcessor
    }
    
    @classmethod
    def create_processor(cls, payment_method: str) -> PaymentProcessor:
        """
        Create appropriate payment processor based on payment method
        
        Args:
            payment_method: Type of payment ('credit_card', 'paypal', etc.)
            
        Returns:
            PaymentProcessor instance
        """
        processor_class = cls._processors.get(payment_method.lower())
        
        if processor_class is None:
            raise ValueError(
                f"Unknown payment method: {payment_method}. "
                f"Available: {list(cls._processors.keys())}"
            )
        
        return processor_class()
    
    @classmethod
    def register_processor(cls, name: str, processor_class):
        """Register new payment processor type"""
        cls._processors[name] = processor_class
    
    @classmethod
    def get_available_methods(cls):
        return list(cls._processors.keys())

# ============ USAGE ============

class CheckoutService:
    """Service that uses the factory to process payments"""
    
    def __init__(self):
        self.factory = PaymentProcessorFactory()
    
    def checkout(self, payment_method: str, amount: float, 
                 currency: str, credentials: Dict):
        """Process a checkout"""
        
        # Create appropriate processor using factory
        processor = self.factory.create_processor(payment_method)
        
        # Validate credentials
        if not processor.validate_credentials(credentials):
            return {'status': 'error', 'message': 'Invalid credentials'}
        
        # Process payment
        result = processor.process_payment(amount, currency)
        return result
    
    def process_refund(self, payment_method: str, transaction_id: str, amount: float):
        """Process a refund"""
        processor = self.factory.create_processor(payment_method)
        return processor.refund_payment(transaction_id, amount)

# Client code
checkout = CheckoutService()

# Process different payment types
print("=== Credit Card Payment ===")
result1 = checkout.checkout(
    payment_method='credit_card',
    amount=99.99,
    currency='USD',
    credentials={'card_number': '1234', 'cvv': '123', 'expiry': '12/25'}
)
print(f"Result: {result1}\n")

print("=== PayPal Payment ===")
result2 = checkout.checkout(
    payment_method='paypal',
    amount=49.99,
    currency='USD',
    credentials={'email': 'user@example.com', 'password': 'secret'}
)
print(f"Result: {result2}\n")

print("=== Crypto Payment ===")
result3 = checkout.checkout(
    payment_method='crypto',
    amount=0.002,
    currency='BTC',
    credentials={'wallet_address': '0x123...', 'private_key': 'key'}
)
print(f"Result: {result3}\n")

# Process refund
print("=== Processing Refund ===")
refund_result = checkout.process_refund('credit_card', 'CC-12345', 99.99)
print(f"Refund: {refund_result}")

# Show available methods
print(f"\nAvailable payment methods: {PaymentProcessorFactory.get_available_methods()}")
```

---

### Example 2: Notification System

```python
from abc import ABC, abstractmethod
from typing import List, Dict

# ============ PRODUCT INTERFACE ============

class Notification(ABC):
    @abstractmethod
    def send(self, recipient: str, message: str, subject: str = None) -> bool:
        pass
    
    @abstractmethod
    def validate_recipient(self, recipient: str) -> bool:
        pass

# ============ CONCRETE PRODUCTS ============

class EmailNotification(Notification):
    def validate_recipient(self, recipient: str) -> bool:
        return '@' in recipient and '.' in recipient
    
    def send(self, recipient: str, message: str, subject: str = None) -> bool:
        if not self.validate_recipient(recipient):
            print(f"Invalid email: {recipient}")
            return False
        
        print(f"📧 EMAIL SENT")
        print(f"   To: {recipient}")
        print(f"   Subject: {subject or 'No Subject'}")
        print(f"   Message: {message}")
        return True

class SMSNotification(Notification):
    def validate_recipient(self, recipient: str) -> bool:
        # Simple phone number validation
        return recipient.replace('+', '').replace('-', '').isdigit()
    
    def send(self, recipient: str, message: str, subject: str = None) -> bool:
        if not self.validate_recipient(recipient):
            print(f"Invalid phone number: {recipient}")
            return False
        
        # SMS has character limit
        if len(message) > 160:
            message = message[:157] + "..."
        
        print(f"📱 SMS SENT")
        print(f"   To: {recipient}")
        print(f"   Message: {message}")
        return True

class PushNotification(Notification):
    def validate_recipient(self, recipient: str) -> bool:
        # Device token validation (simplified)
        return len(recipient) > 20
    
    def send(self, recipient: str, message: str, subject: str = None) -> bool:
        if not self.validate_recipient(recipient):
            print(f"Invalid device token: {recipient}")
            return False
        
        print(f"🔔 PUSH NOTIFICATION SENT")
        print(f"   Device: {recipient[:20]}...")
        print(f"   Title: {subject or 'Notification'}")
        print(f"   Message: {message}")
        return True

class SlackNotification(Notification):
    def validate_recipient(self, recipient: str) -> bool:
        # Slack channel validation
        return recipient.startswith('#') or recipient.startswith('@')
    
    def send(self, recipient: str, message: str, subject: str = None) -> bool:
        if not self.validate_recipient(recipient):
            print(f"Invalid Slack recipient: {recipient}")
            return False
        
        print(f"💬 SLACK MESSAGE SENT")
        print(f"   Channel: {recipient}")
        print(f"   Message: {message}")
        return True

# ============ FACTORY ============

class NotificationFactory:
    """Factory to create notification senders"""
    
    _notification_types = {
        'email': EmailNotification,
        'sms': SMSNotification,
        'push': PushNotification,
        'slack': SlackNotification
    }
    
    @classmethod
    def create_notification(cls, notification_type: str) -> Notification:
        """Create notification sender"""
        notification_class = cls._notification_types.get(notification_type.lower())
        
        if notification_class is None:
            raise ValueError(
                f"Unknown notification type: {notification_type}. "
                f"Available: {list(cls._notification_types.keys())}"
            )
        
        return notification_class()
    
    @classmethod
    def create_multiple(cls, notification_types: List[str]) -> List[Notification]:
        """Create multiple notification senders"""
        return [cls.create_notification(nt) for nt in notification_types]

# ============ USAGE ============

class NotificationService:
    """Service that sends notifications via multiple channels"""
    
    def __init__(self):
        self.factory = NotificationFactory()
    
    def send_notification(self, channel: str, recipient: str, 
                         message: str, subject: str = None) -> bool:
        """Send notification via single channel"""
        notifier = self.factory.create_notification(channel)
        return notifier.send(recipient, message, subject)
    
    def broadcast(self, channels: List[str], recipients: Dict[str, str],
                 message: str, subject: str = None):
        """Send notification via multiple channels"""
        results = {}
        
        for channel in channels:
            recipient = recipients.get(channel)
            if recipient:
                notifier = self.factory.create_notification(channel)
                results[channel] = notifier.send(recipient, message, subject)
            else:
                print(f"⚠️  No recipient for {channel}")
                results[channel] = False
        
        return results

# Client code
service = NotificationService()

print("=== Single Channel Notification ===")
service.send_notification(
    channel='email',
    recipient='user@example.com',
    message='Your order has been shipped!',
    subject='Order Update'
)

print("\n=== Multi-Channel Broadcast ===")
results = service.broadcast(
    channels=['email', 'sms', 'push'],
    recipients={
        'email': 'user@example.com',
        'sms': '+1-555-0123',
        'push': 'device_token_abc123xyz789_long_string'
    },
    message='Your account has been verified!',
    subject='Account Verification'
)

print(f"\nBroadcast results: {results}")

print("\n=== Slack Notification ===")
service.send_notification(
    channel='slack',
    recipient='#engineering',
    message='Deploy completed successfully! 🚀'
)
```

---

### Example 3: Database Connection Factory

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List

# ============ PRODUCT INTERFACE ============

class DatabaseConnection(ABC):
    @abstractmethod
    def connect(self, connection_string: str) -> bool:
        pass
    
    @abstractmethod
    def execute_query(self, query: str) -> List[Dict]:
        pass
    
    @abstractmethod
    def execute_command(self, command: str) -> bool:
        pass
    
    @abstractmethod
    def close(self) -> bool:
        pass

# ============ CONCRETE PRODUCTS ============

class PostgreSQLConnection(DatabaseConnection):
    def __init__(self):
        self.connection = None
    
    def connect(self, connection_string: str) -> bool:
        print(f"🐘 Connecting to PostgreSQL: {connection_string}")
        self.connection = "PostgreSQL Connection Object"
        return True
    
    def execute_query(self, query: str) -> List[Dict]:
        print(f"PostgreSQL Query: {query}")
        # Simulated result
        return [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]
    
    def execute_command(self, command: str) -> bool:
        print(f"PostgreSQL Command: {command}")
        return True
    
    def close(self) -> bool:
        print("Closing PostgreSQL connection")
        self.connection = None
        return True

class MySQLConnection(DatabaseConnection):
    def __init__(self):
        self.connection = None
    
    def connect(self, connection_string: str) -> bool:
        print(f"🐬 Connecting to MySQL: {connection_string}")
        self.connection = "MySQL Connection Object"
        return True
    
    def execute_query(self, query: str) -> List[Dict]:
        print(f"MySQL Query: {query}")
        return [{'id': 1, 'name': 'Charlie'}, {'id': 2, 'name': 'Diana'}]
    
    def execute_command(self, command: str) -> bool:
        print(f"MySQL Command: {command}")
        return True
    
    def close(self) -> bool:
        print("Closing MySQL connection")
        self.connection = None
        return True

class MongoDBConnection(DatabaseConnection):
    def __init__(self):
        self.connection = None
    
    def connect(self, connection_string: str) -> bool:
        print(f"🍃 Connecting to MongoDB: {connection_string}")
        self.connection = "MongoDB Connection Object"
        return True
    
    def execute_query(self, query: str) -> List[Dict]:
        print(f"MongoDB Query: {query}")
        return [{'_id': '1', 'name': 'Eve'}, {'_id': '2', 'name': 'Frank'}]
    
    def execute_command(self, command: str) -> bool:
        print(f"MongoDB Command: {command}")
        return True
    
    def close(self) -> bool:
        print("Closing MongoDB connection")
        self.connection = None
        return True

class SQLiteConnection(DatabaseConnection):
    def __init__(self):
        self.connection = None
    
    def connect(self, connection_string: str) -> bool:
        print(f"💾 Connecting to SQLite: {connection_string}")
        self.connection = "SQLite Connection Object"
        return True
    
    def execute_query(self, query: str) -> List[Dict]:
        print(f"SQLite Query: {query}")
        return [{'id': 1, 'name': 'Grace'}, {'id': 2, 'name': 'Henry'}]
    
    def execute_command(self, command: str) -> bool:
        print(f"SQLite Command: {command}")
        return True
    
    def close(self) -> bool:
        print("Closing SQLite connection")
        self.connection = None
        return True

# ============ FACTORY ============

class DatabaseFactory:
    """Factory to create database connections"""
    
    _db_types = {
        'postgresql': PostgreSQLConnection,
        'postgres': PostgreSQLConnection,
        'mysql': MySQLConnection,
        'mongodb': MongoDBConnection,
        'mongo': MongoDBConnection,
        'sqlite': SQLiteConnection
    }
    
    @classmethod
    def create_connection(cls, db_type: str) -> DatabaseConnection:
        """Create database connection"""
        db_class = cls._db_types.get(db_type.lower())
        
        if db_class is None:
            raise ValueError(
                f"Unknown database type: {db_type}. "
                f"Available: {set(cls._db_types.values())}"
            )
        
        return db_class()
    
    @classmethod
    def create_from_config(cls, config: Dict[str, str]) -> DatabaseConnection:
        """Create database connection from configuration"""
        db_type = config.get('type')
        connection_string = config.get('connection_string')
        
        if not db_type:
            raise ValueError("Database type not specified in config")
        
        db = cls.create_connection(db_type)
        
        if connection_string:
            db.connect(connection_string)
        
        return db

# ============ USAGE ============

class DataRepository:
    """Repository that uses database factory"""
    
    def __init__(self, db_type: str, connection_string: str):
        self.db = DatabaseFactory.create_connection(db_type)
        self.db.connect(connection_string)
    
    def get_users(self) -> List[Dict]:
        return self.db.execute_query("SELECT * FROM users")
    
    def create_user(self, name: str) -> bool:
        return self.db.execute_command(f"INSERT INTO users (name) VALUES ('{name}')")
    
    def cleanup(self):
        self.db.close()

# Client code
print("=== PostgreSQL Repository ===")
pg_repo = DataRepository('postgresql', 'postgresql://localhost:5432/mydb')
users = pg_repo.get_users()
print(f"Users: {users}")
pg_repo.create_user('New User')
pg_repo.cleanup()

print("\n=== MongoDB Repository ===")
mongo_repo = DataRepository('mongodb', 'mongodb://localhost:27017/mydb')
users = mongo_repo.get_users()
print(f"Users: {users}")
mongo_repo.cleanup()

print("\n=== Using Config ===")
config = {
    'type': 'mysql',
    'connection_string': 'mysql://localhost:3306/mydb'
}
db = DatabaseFactory.create_from_config(config)
result = db.execute_query("SELECT * FROM products")
print(f"Products: {result}")
db.close()
```

---

## Common Pitfalls

### ❌ Pitfall 1: Factory Returns Wrong Type

```python
# BAD - Returns string instead of object
class BadFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return "dog"  # Wrong! Should return Dog instance
        elif animal_type == "cat":
            return "cat"

# GOOD - Returns proper object
class GoodFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()  # Correct!
        elif animal_type == "cat":
            return Cat()
```

---

### ❌ Pitfall 2: Not Using Polymorphism

```python
# BAD - Client code has to know specific types
def process_payment(payment_type, amount):
    if payment_type == "credit_card":
        processor = CreditCardProcessor()
        processor.process_credit_card(amount)
    elif payment_type == "paypal":
        processor = PayPalProcessor()
        processor.process_paypal_payment(amount)
    # Different methods for each type!

# GOOD - All types implement same interface
def process_payment(payment_type, amount):
    processor = PaymentFactory.create_processor(payment_type)
    processor.process_payment(amount)  # Same method for all!
```

---

### ❌ Pitfall 3: Hardcoded Factory Logic

```python
# BAD - Adding new type requires modifying factory
class BadFactory:
    def create(self, type_name):
        if type_name == "A":
            return TypeA()
        elif type_name == "B":
            return TypeB()
        # Must add elif for every new type!

# GOOD - Use registry pattern
class GoodFactory:
    _registry = {}
    
    @classmethod
    def register(cls, name, type_class):
        cls._registry[name] = type_class
    
    @classmethod
    def create(cls, name):
        return cls._registry[name]()

# Register types
GoodFactory.register("A", TypeA)
GoodFactory.register("B", TypeB)
# Can add new types without modifying factory!
```

---

## Best Practices

### ✅ 1. Use Type Hints

```python
from typing import Protocol

class Animal(Protocol):
    def speak(self) -> str: ...

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type: str) -> Animal:
        # Type hints make intent clear
        pass
```

---

### ✅ 2. Provide Clear Error Messages

```python
class Factory:
    def create(self, type_name: str):
        if type_name not in self._types:
            raise ValueError(
                f"Unknown type: {type_name}. "
                f"Available types: {list(self._types.keys())}"
            )
```

---

### ✅ 3. Use Registry for Extensibility

```python
class ExtensibleFactory:
    _registry = {}
    
    @classmethod
    def register(cls, name, product_class):
        """Allow external code to add new products"""
        cls._registry[name] = product_class
```

---

### ✅ 4. Document Factory Behavior

```python
class PaymentFactory:
    """
    Factory for creating payment processors.
    
    Supported payment methods:
    - 'credit_card': Credit card processing
    - 'paypal': PayPal processing
    - 'crypto': Cryptocurrency processing
    
    Example:
        processor = PaymentFactory.create('credit_card')
        result = processor.process_payment(100.00, 'USD')
    """
    pass
```

---

## Summary Comparison

| Pattern Type | Use When | Complexity | Example |
|-------------|----------|------------|---------|
| **Simple Factory** | Single factory creates variants | Low | AnimalFactory creates Dog/Cat |
| **Factory Method** | Subclasses decide what to create | Medium | PDFApp creates PDFDoc |
| **Abstract Factory** | Need families of related objects | High | WindowsFactory creates Windows UI family |
