from design_patterns.structural_patterns.decorator.decorators.abs_decorator import AbstractDecorator


class Black(AbstractDecorator):

    @property
    def description(self):
        return self.car.description + ', black'

    @property
    def cost(self):
        return self.car.cost + 1800.00
