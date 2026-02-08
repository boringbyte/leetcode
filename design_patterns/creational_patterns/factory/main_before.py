from design_patterns.creational_patterns.factory.autos.chevyvolt import ChevyVolt
from design_patterns.creational_patterns.factory.autos.fordfusion import FordFusion
from design_patterns.creational_patterns.factory.autos.jeepsahara import JeepSahara
from design_patterns.creational_patterns.factory.autos.nullcar import NullCar


class VehicleFactory:

    @staticmethod
    def create_vehicle(carname):
        if carname == 'Chevy':
            return ChevyVolt(carname)
        elif carname == 'Ford':
            return FordFusion(carname)
        elif carname == 'Jeep':
            return JeepSahara(carname)
        else:
            return NullCar(carname)


if __name__ == '__main__':
    # 1. Adding a different car, needs opening the old code, and it breaks open-closed principle
    # 2. Above, we are directly instantiating the car classes, and it breaks dependency inversion principle,
    #    since we are depending on the implementation of those classes.
    factory = VehicleFactory()
    for carname in ['Chevy', 'Ford', 'Jeep', 'Tesla']:
        car = factory.create_vehicle(carname)
        car.start()
        car.stop()
