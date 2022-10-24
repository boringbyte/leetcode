from abs_composite import AbstractComposite


class Person(AbstractComposite):

    def __init__(self, name, birthdate):
        self._name = name
        self._birthdate = birthdate

    def get_oldest(self):
        return self

    @property
    def name(self):
        return self._name

    @property
    def birthdate(self):
        return self._birthdate
