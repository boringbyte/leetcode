from design_patterns.structural_patterns.decorator.decorators.abs_decorator import AbstractDecorator


class Leather(AbstractDecorator):

    @property
    def description(self):
        return self.car.description + ', top-grain leather'

    @property
    def cost(self):
        return self.car.cost + 2700.00
