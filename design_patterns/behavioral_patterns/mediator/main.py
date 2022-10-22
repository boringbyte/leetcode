import random
from ped_mediator import PetMediator
from pets.cat import Cat
from pets.dog import Dog
from pets.fish import Fish


def main():
    cat = Cat('Schrodinger')
    dog = Dog('Pluto')
    fish = Fish('Wanda')

    pm = PetMediator(cat, dog, fish)
    cat.mediator = pm
    dog.mediator = pm
    fish.mediator = pm

    t = random.randint(-1, 2)
    pm.time_of_day(t)


if __name__ == '__main__':
    main()
