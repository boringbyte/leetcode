import threading


class SingletonMeta(type):

    _instance = {}
    _instance_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance:
            with cls._instance_lock:
                if cls not in cls._instance:
                    instance = super().__call__(*args, **kwargs)
                    cls._instance[cls] = instance
        return cls._instance[cls]


class Singleton(metaclass=SingletonMeta):

    def __init__(self):
        self.value = None


if __name__ == '__main__':
    s1 = Singleton()
    s2 = Singleton()
    print(s1 is s2)
