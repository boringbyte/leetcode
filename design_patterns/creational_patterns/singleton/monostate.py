class MonoState(object):
    _state = {}  # Monostate contains the dictionary containing the single state for all the instances

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        self.__dict__ = cls._state  # A dict object where the instance state is stored
        return self                 # is redirected to single state dictionary

