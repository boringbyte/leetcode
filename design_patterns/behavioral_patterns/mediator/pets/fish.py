import random
from design_patterns.behavioral_patterns.mediator.pets.abs_pet import AbstractPet


class Fish(AbstractPet):

    def __init__(self, name):
        self.name = name
        self.mediator = None

    def needs_food(self):
        if self.mediator.is_morning():
            print(f'Feed {self.name}')
        else:
            print(f'{self.name} is not hungry')

    def is_alive(self):
        return random.randint(0, 1)
