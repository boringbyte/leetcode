from .abs_factory import AbstractFactory
from design_patterns.creational_patterns.factory.autos import FordFusion


class FordFactory(AbstractFactory):

    def create_auto(self):
        self.ford = ford = FordFusion()
        ford.name = 'Ford Fusion'
        return ford
