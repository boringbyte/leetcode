from design_patterns.creational_patterns.abstract_factory.factories.ford_factory import FordFactory
from design_patterns.creational_patterns.abstract_factory.factories.gm_factory import GMFactory


if __name__ == '__main__':
    """
    1. Encapsulates object instantiation
    2. Supports dependency inversion
    3. Clients can write to an abstraction
    4. Factory vs Abstract Factory?
        i. Factory is great when you don't know which concrete classes you'll need
        ii. Abstract Factory is useful when you have families of objects
    """
    for factory in FordFactory, GMFactory:
        car = factory.create_economy()
        car.start()
        car.stop()
        car = factory.create_sport()
        car.start()
        car.stop()
        car = factory.create_luxury()
        car.start()
        car.stop()
