from design_patterns.creational_patterns.prototype.abs_prototype import AbstractPrototype
from design_patterns.creational_patterns.prototype.abs_computer import AbstractComputer
import copy


class MainBoard(object):
    manufacturer: str
    model: str

    def __init__(self, manufacturer, model):
        self.manufacturer = manufacturer
        self.model = model


class Tower(AbstractComputer, AbstractPrototype):

    def __init__(self, model, mainboard, processor, memory, hard_drive, graphics, monitor):
        self.model = model
        self.mainboard = mainboard
        self.processor = processor
        self.memory = memory
        self.hard_drive = hard_drive
        self.graphics = graphics
        self.monitor = monitor

    def display(self):
        print('Custom Computer:' + self.model)
        print(f'\t{"Mainboard":>12}: {self.mainboard.model}')
        print(f'\t{"Processor":>12}: {self.processor}')
        print(f'\t{"Memory":>12}: {self.memory}')
        print(f'\t{"Hard Drive":>12}: {self.hard_drive}')
        print(f'\t{"Graphics":>12}: {self.graphics}')
        print(f'\t{"Monitor":>12}: {self.monitor if self.monitor else "None"}')

    def clone(self):
        return copy.deepcopy(self)
