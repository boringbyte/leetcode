from design_patterns.creational_patterns.abstract_factory.autos.abs_auto import AbstractAuto


class CadillacCTS(AbstractAuto):

    def start(self):
        print('Cadillac CTS running smoothly')

    def stop(self):
        print('Cadillac CTS shutting down.')
