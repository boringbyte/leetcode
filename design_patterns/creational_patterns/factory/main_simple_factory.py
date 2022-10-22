from design_patterns.creational_patterns.factory.autofactory import AutoFactory


if __name__ == '__main__':
    """
    1. We solved open-closed violation. We can simply add new automobiles as new class to autos modules and add it to 
       __init__ file
    2. We eliminated the dependency on the implementation of the automobile classes. 
       The main program only knows that those classes implement the abstract methods from the abstract base class.
    3. We also separated the main program and auto factory loader.
    4. We are limited to one factory. This can be solved by classic factory method pattern.
        - There is abstract product class
        - There is concrete product class
        - There is abstract factory class which has abstract create_product() method
        - There is concrete factory class which implements create_product() method and return concrete product object.    
    """
    factory = AutoFactory()
    for carname in ['ChevyVolt', 'FordFusion', 'JeepSahara', 'Tesla']:
        car = factory.create_instance(carname)
        car.start()
        car.stop()
