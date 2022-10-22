import random
from design_patterns.behavioral_patterns.mediator.pets.abs_pet import AbstractPet


class Cat(AbstractPet):
    is_awake = random.randint(0, 1)

    def wants_out(self):
        if self.mediator.is_fish_alive():
            print(f'Let {self.name} in')
        else:
            print(f'Let {self.name} out')

    def is_asleep(self):
        return not self.is_awake
