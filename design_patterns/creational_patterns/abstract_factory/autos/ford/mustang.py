from design_patterns.creational_patterns.abstract_factory.autos.abs_auto import AbstractAuto


class FordMustang(AbstractAuto):

    def start(self):
        print('Ford Mustang roaring and ready to go!')

    def stop(self):
        print('Ford Mustang shutting down.')
