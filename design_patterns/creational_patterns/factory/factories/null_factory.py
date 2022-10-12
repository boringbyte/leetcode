from .abs_factory import AbstractFactory
from design_patterns.creational_patterns.factory.autos import NullCar


class NullFactory(AbstractFactory):

    def create_auto(self):
        self.jeep = jeep = NullCar()
        jeep.name = 'Jeep Sahara'
        return jeep
