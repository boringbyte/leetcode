from design_patterns.creational_patterns.abstract_factory.autos.abs_auto import AbstractAuto


class ChevySpark(AbstractAuto):

    def start(self):
        print('Chevy Spark running cheaply')

    def stop(self):
        print('Chevy Spark shutting down.')
