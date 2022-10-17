from design_patterns.structural_patterns.decorator.cars.abs_car import AbstractCar


class Sport(AbstractCar):

    @property
    def description(self):
        return 'Sport'

    @property
    def cost(self):
        return 15000.00
