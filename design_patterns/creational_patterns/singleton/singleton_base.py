class Singleton(object):
    _instances = {}  # dict([cls, instance])

    def __new__(cls, *args, **kwargs):  # gets invoked even before __init__
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
        return cls._instances[cls]
