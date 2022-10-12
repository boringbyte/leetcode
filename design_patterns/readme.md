## SOLID Principles of Object-oriented Design

- S - Single Responsibility:
  - A class should have only one responsibility.
- O - Open-closed:
  - A class should be open for extension usually by inheritance, but closed for modification.
- L - Liskov substitution:
  - Subclasses should be stand in for their parents without breaking anything.
- I - Interface segregation:
  - A specific interfaces are better than having one do-all interface.
- D - Dependency inversion:
  - We should program towards abstraction, but not implementations. Implementations can vary but not abstractions.


## Main Classification of Design Patterns
- Creational Patterns:
  - Factory Method
    - Defines interface for creating an object
    - Lets subclasses decide which object
    - Defers instantiation of subclasses
    - Also known as Virtual Constructor
    - If we are using if else for creating classes, then we can replace that with this pattern.
  - Abstract Factory
  - Builder
  - Prototype
  - Singleton
- Structural Patterns:
  - Adapter
  - Bridge
  - Composite
  - Decorator
  - Facade
  - Flyweight
  - Proxy
- Behavioral Patterns:
  - Strategy
  - Command
  - State
  - Observer
  - Visitor
  - Chain of Responsibility
  - Mediator
  - Memento
  - Null
  - Template
  - Interpreter