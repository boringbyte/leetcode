from design_patterns.creational_patterns.factory.autos.abs_auto import AbstractAuto


class ChevyVolt(AbstractAuto):

    def start(self):
        print('Chevrolet Volt running with shocking power!')

    def stop(self):
        print('Chevrolet Volt shutting down.')
