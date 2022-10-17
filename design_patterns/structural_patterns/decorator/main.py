from cars.economy import Economy
from decorators.v6 import V6
from decorators.vinyl import Vinyl
from decorators.black import Black


def main():
    car = Economy()
    show(car)
    car = V6(car)
    show(car)
    car = Vinyl(car)
    show(car)
    car = Black(car)
    show(car)


def show(car):
    print(f'Description: {car.description}; cost: ${car.cost}')


if __name__ == '__main__':
    """
    1. Much more flexible than static inheritance
    2. Keeps things simple. Each decorator is need to be concerted with only its specific attributes and object to decorate
    3. No practical limit to decorations.
    4. Transparent to client.
    5. A decorator has a different type.
    6. Many little objects for each decoration.
    """
    main()
