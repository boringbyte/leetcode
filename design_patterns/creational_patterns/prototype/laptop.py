from design_patterns.creational_patterns.prototype.abs_prototype import AbstractPrototype
from design_patterns.creational_patterns.prototype.abs_computer import AbstractComputer
import copy


class Laptop(AbstractComputer, AbstractPrototype):

    def __init__(self, model, processor, memory, hard_drive, graphics, screen):
        self.model = model
        self.processor = processor
        self.memory = memory
        self.hard_drive = hard_drive
        self.graphics = graphics
        self.screen = screen

    def display(self):
        print('Custom Computer:' + self.model)
        print(f'\t{"Processor":>10}: {self.processor}')
        print(f'\t{"Memory":>10}: {self.memory}')
        print(f'\t{"Hard Drive":>10}: {self.hard_drive}')
        print(f'\t{"Graphics":>10}: {self.graphics}')
        print(f'\t{"Screen":>10}: {self.screen}')

    def clone(self):
        return copy.copy(self)
