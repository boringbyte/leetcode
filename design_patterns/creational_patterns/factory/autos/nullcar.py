from design_patterns.creational_patterns.factory.autos.abs_auto import AbstractAuto


class NullCar(AbstractAuto):

    def __init__(self, carname):
        self._carname = carname

    def start(self):
        print(f'Unknown car "{self._carname}".')

    def stop(self):
       pass
