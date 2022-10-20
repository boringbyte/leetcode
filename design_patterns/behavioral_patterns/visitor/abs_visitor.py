from abc import ABC, abstractmethod


class Visitor(ABC):

    @abstractmethod
    def visit_person(self, person):
        pass

    @abstractmethod
    def visit_tree(self, tree):
        pass
