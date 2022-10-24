from . import PROVIDER
from facade_factory import FacadeFactory


def main():
    facade = FacadeFactory.create_facade(PROVIDER)
    facade.get_employees()


if __name__ == '__main__':
    """
        1. Shields clients from subsystem details
        2. Reduces the objects to handle
        3. Promotes weak coupling
        4. Vary the subsystem
        5. No change to client code
        6. Clients can still use subsystems directly
    """
    main()
