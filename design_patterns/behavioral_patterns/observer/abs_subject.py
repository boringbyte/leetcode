from abc import ABC, abstractmethod
from abs_observer import AbstractObserver


class AbstractSubject(ABC):

    _observers = set()

    def attach(self, observer):
        if not isinstance(observer, AbstractObserver):
            raise TypeError('Observer not derived from AbstractObserver class')
        self._observers |= {observer}

    def detach(self, observer):
        self._observers -= {observer}

    def notify(self, value=None):
        for observer in self._observers:
            if value is None:
                observer.update()
            else:
                observer.update(value)
