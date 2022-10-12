from .abs_factory import AbstractFactory
from design_patterns.creational_patterns.factory.autos import ChevyVolt


class ChevyFactory(AbstractFactory):

    def create_auto(self):
        self.chevy = chevy = ChevyVolt()
        chevy.name = 'Chevy Volt'
        return chevy
