from .abs_factory import AbstractFactory
from design_patterns.creational_patterns.abstract_factory.autos.ford.fiesta import FordFiesta
from design_patterns.creational_patterns.abstract_factory.autos.ford.mustang import FordMustang
from design_patterns.creational_patterns.abstract_factory.autos.ford.lincoln import LincolnMKS


class FordFactory(AbstractFactory):

    @staticmethod
    def create_economy():
        return FordFiesta()

    @staticmethod
    def create_sport():
        return FordMustang()

    @staticmethod
    def create_luxury():
        return LincolnMKS()
