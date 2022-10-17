from design_patterns.structural_patterns.decorator.cars.abs_car import AbstractCar


class Economy(AbstractCar):

    @property
    def description(self):
        return 'Economy'

    @property
    def cost(self):
        return 12000.00
