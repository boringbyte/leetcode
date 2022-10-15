from collections import Iterable
from functools import reduce
from datetime import date
from abs_composite import AbstractComposite


class Tree(Iterable, AbstractComposite):

    def __init__(self, members):
        self._members = members  # member can be either Person or another composite tree

    def __iter__(self):
        return iter(self._members)

    def get_oldest(self):
        def f(t1, t2):
            t1_, t2_ = t1.get_oldest(), t2.get_oldest()
            return t1_ if t1_.birthdate < t2_.birthdate else t2_
        return reduce(f, self, NullPerson())  # Recursive DFS


class NullPerson(AbstractComposite):
    name = None
    birthdate = date.max

    def get_oldest(self):
        return self
