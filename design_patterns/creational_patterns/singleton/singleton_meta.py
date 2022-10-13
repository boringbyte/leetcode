"""
    metaclass is a class's class
    class is an instance of a metaclass
    control building of class
"""

import datetime


class Singleton(type):  # Here inherits from type and not object
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
