from design_patterns.creational_patterns.abstract_factory.autos.abs_auto import AbstractAuto


class FordFiesta(AbstractAuto):

    def start(self):
        print('Ford Fiesta running cheaply')

    def stop(self):
        print('Ford Fiesta shutting down.')
