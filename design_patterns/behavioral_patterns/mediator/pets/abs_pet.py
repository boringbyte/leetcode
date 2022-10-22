from abc import ABC, abstractmethod


class AbstractPet(ABC):

    def __init__(self, name):
        self.name = name
        self.mediator = None
