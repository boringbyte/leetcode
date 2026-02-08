class Singleton:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


if __name__ == '__main__':
    s1 = Singleton()
    s2 = Singleton()
    print(s1 is s2)
