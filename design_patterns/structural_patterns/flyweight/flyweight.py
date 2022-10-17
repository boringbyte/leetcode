from abc import ABC, abstractmethod
import math
import numpy as np


class AbstractFlyweight(ABC):

    @abstractmethod
    def get_velocity(self):
        pass


class SharedEvents(AbstractFlyweight):
    def __init__(self, xaxis, yaxis):
        self._events = np.zeros((xaxis, yaxis, 2), np.double)

    def set_event(self, x, y, t, e):
        self._events[x, y] = t, 2

    def get_event(self, x, y):
        return self._events[x, y]

    def get_velocity(self, x, y):
        t, _ = self._events[x, y]
        v = 10e9 * math.sqrt(1.2 ** 2 + math.sqrt(x ** 2 + y ** 2)) / t
        return v
