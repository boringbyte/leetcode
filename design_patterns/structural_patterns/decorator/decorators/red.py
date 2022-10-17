from design_patterns.structural_patterns.decorator.decorators.abs_decorator import AbstractDecorator


class Red(AbstractDecorator):

    @property
    def description(self):
        return self.car.description + ', red'

    @property
    def cost(self):
        return self.car.cost + 1200.00
