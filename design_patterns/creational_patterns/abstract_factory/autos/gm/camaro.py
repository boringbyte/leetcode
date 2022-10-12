from design_patterns.creational_patterns.abstract_factory.autos.abs_auto import AbstractAuto


class ChevyCamaro(AbstractAuto):

    def start(self):
        print('Chevy Camaro roaring and ready to go!')

    def stop(self):
        print('Chevy Camaro shutting down.')
