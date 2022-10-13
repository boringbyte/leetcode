from abs_prototype import AbstractPrototype


class PrototypeManager(dict):

    def __setitem__(self, key, prototype):
        if issubclass(prototype, AbstractPrototype):
            dict.__setitem__(self, key, prototype)
