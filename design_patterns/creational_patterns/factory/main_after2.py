from design_patterns.creational_patterns.factory.factories import loader


if __name__ == '__main__':
    """
    1. Added an abstract factory base class
    2. Many factories can be implemented
    3. The implementation can vary
    4. A complex factory might use other patterns
    
    """
    for factory_name in ['chevy_factory', 'jeep_factory', 'ford_factory', 'tesla_factory']:
        factory = loader.load_factory(factory_name)
        car = factory.create_auto()
        car.start()
        car.stop()
