from .abs_factory import AbstractFactory
from design_patterns.creational_patterns.abstract_factory.autos.gm.spark import ChevySpark
from design_patterns.creational_patterns.abstract_factory.autos.gm.camaro import ChevyCamaro
from design_patterns.creational_patterns.abstract_factory.autos.gm.cadillac import CadillacCTS


class GMFactory(AbstractFactory):

    @staticmethod
    def create_economy():
        return ChevySpark()

    @staticmethod
    def create_sport():
        return ChevyCamaro()

    @staticmethod
    def create_luxury():
        return CadillacCTS()
