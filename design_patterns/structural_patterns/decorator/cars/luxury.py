from design_patterns.structural_patterns.decorator.cars.abs_car import AbstractCar


class Luxury(AbstractCar):

    @property
    def description(self):
        return 'Luxury'

    @property
    def cost(self):
        return 18000.00
