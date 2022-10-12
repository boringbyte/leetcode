from design_patterns.creational_patterns.factory.autos.abs_auto import AbstractAuto


class FordFusion(AbstractAuto):

    def start(self):
        print('Cool Ford Fusion running smoothly')

    def stop(self):
        print('Ford Fusion shutting down.')
