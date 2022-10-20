from abs_tree import AbstractTree


class NullPerson(AbstractTree):

    def __init__(self):
        pass

    @property
    def name(self):
        pass

    @property
    def birthdate(self):
        pass

    def accept(self, visitor):
        visitor.visit_person(self)
