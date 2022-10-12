from .abs_factory import AbstractFactory
from design_patterns.creational_patterns.factory.autos import JeepSahara


class JeepFactory(AbstractFactory):

    def create_auto(self):
        self.jeep = jeep = JeepSahara()
        jeep.name = 'Jeep Sahara'
        return jeep
