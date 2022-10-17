from design_patterns.structural_patterns.decorator.cars.abs_car import AbstractCar


class AbstractDecorator(AbstractCar):

    def __init__(self, car):
        self._car = car

    @property
    def car(self):
        return self._car
