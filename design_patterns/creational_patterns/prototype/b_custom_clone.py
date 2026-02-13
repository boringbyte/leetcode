from abc import ABC, abstractmethod


class Shape(ABC):

    def __init__(self, color, x, y):
        self.color = color
        self.x = x
        self.y = y

    @abstractmethod
    def clone(self):
        pass

    @abstractmethod
    def draw(self):
        pass


class Circle(Shape):

    def __init__(self, color, x, y, radius):
        super().__init__(color, x, y)
        self.radius = radius

    def clone(self):
        return Circle(self.color, self.x, self.y, self.radius)

    def draw(self):
        return f"Circle(color={self.color}, pos=({self.x},{self.y}), radius={self.radius})"


class Rectangle(Shape):

    def __init__(self, color, x, y, width, height):
        super().__init__(color, x, y)
        self.width = width
        self.height = height

    def clone(self):
        return Rectangle(self.color, self.x, self.y, self.width, self.height)

    def draw(self):
        return f"Rectangle(color={self.color}, pos=({self.x},{self.y}), size={self.width}x{self.height})"


if __name__ == '__main__':
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
