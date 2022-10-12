from abc import ABC, abstractmethod


class MyABC(ABC):
    """Abstract Base Class definition"""

    @abstractmethod
    def do_something(self, value):
        """Required method"""

    @property
    @abstractmethod
    def price(self):
        """Required price"""
