import copy

class Prototype:

    def clone(self):
        return copy.deepcopy(self)


class PrototypeRegistry:

    def __init__(self):
        self._prototypes: dict[str, Prototype] = {}

    def register(self,name: str, prototype: Prototype):
        self._prototypes[name] = prototype
        print(f"Registered prototype: {name}")

    def unregister(self, name: str):
        if name in self._prototypes:
            del self._prototypes[name]
            print(f"Unregistered prototype: {name}")

    def clone(self, name: str):
        prototype = self._prototypes.get(name)
        if prototype is None:
            raise ValueError(f"Protype '{name}' not found in registry")
        return prototype.clone()

    def list_prototypes(self):
        return list(self._prototypes.keys())


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


if __name__ == '__main__':
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
