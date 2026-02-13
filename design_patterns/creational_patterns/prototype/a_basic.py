import copy
from abc import ABC, abstractmethod


class Prototype(ABC):

    @abstractmethod
    def clone(self):
        pass


class ConcretePrototype(Prototype):

    def __init__(self, name, value, items=None):
        self.name = name
        self.value = value
        self.items = items if items is not None else []

    def clone(self):
        return copy.deepcopy(self)

    def __str__(self):
        return f"ConcreteProtype(name={self.name}, value={self.value}, items={self.items})"


if __name__ == '__main__':
    original = ConcretePrototype("Original", 100, ["item1", "item2"])
    print("Original: ", original)

    clone1 = original.clone()
    clone1.name = "Clone 1"
    clone1.value = 200
    clone1.items.append("item3")

    print("\nAfter modifying clone:")
    print("Original:", original)
    print("Clone 1:", clone1)

    clone2 = original.clone()
    clone2.name = "Clone2"
    print("Clone 2:", clone2)
