from design_patterns.creational_patterns.factory.autos.abs_auto import AbstractAuto


class JeepSahara(AbstractAuto):

    def start(self):
        print('Jeep Sahara running ruggedly.')

    def stop(self):
        print('Jeep Sahara shutting down.')
