from abc import ABC, abstractmethod


class Discount(ABC):
    @property
    @abstractmethod
    def discount(self):
        pass


class StudentDiscount(Discount):
    @property
    def discount(self):
        return 10


class CorporateDiscount(Discount):
    @property
    def discount(self):
        return 20


class NoDiscount(Discount):
    @property
    def discount(self):
        return 0
