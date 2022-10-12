from design_patterns.creational_patterns.factory.autos.chevyvolt import ChevyVolt
from design_patterns.creational_patterns.factory.autos.fordfusion import FordFusion
from design_patterns.creational_patterns.factory.autos.jeepsahara import JeepSahara
from design_patterns.creational_patterns.factory.autos.nullcar import NullCar


def get_car(carname):
    if carname == 'Chevy':
        return ChevyVolt()
    elif carname == 'Ford':
        return FordFusion()
    elif carname == 'Jeep':
        return JeepSahara()
    else:
        return NullCar(carname)


if __name__ == '__main__':
    # 1. Adding a different car, needs opening the old code, and it breaks open-closed principle
    # 2. Above, we are directly instantiating the car classes, and it breaks dependency inversion principle,
    #    since we are depending on the implementation of those classes.
    for carname in ['Chevy', 'Ford', 'Jeep', 'Tesla']:
        car = get_car(carname)
        car.start()
        car.stop()
