from design_patterns.behavioral_patterns.mediator.pets.abs_pet import AbstractPet


class Dog(AbstractPet):

    def __init__(self, name):
        self.name = name
        self.mediator = None

    def wants_walk(self):
        if self.mediator.is_cat_asleep():
            print(f'Walk {self.name}')
        else:
            print(f'Wake up {self.name}')
            self.mediator.wake_up_cat()
