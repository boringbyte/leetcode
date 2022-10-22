from bus import Bus
from airplane import Airplane


def main():
    travel('New York', Bus)
    travel('Amsterdam', Airplane)


def travel(destination, transport):
    print(f'\nTaking the {transport.__name__} to {destination} =========>')

    means = transport(destination)
    means.take_trip()


if __name__ == '__main__':
    """
        - Technique to reuse code.
        - Parent class calls the subclass operations.
        - Don't call us, we'll call you
        - Operations can be in ABC
        - Hook operations can inject special logic
    """
    main()
