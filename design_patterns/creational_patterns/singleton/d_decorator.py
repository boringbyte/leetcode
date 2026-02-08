import threading

def singleton(cls):
    instances = {}
    instance_lock = threading.Lock()

    def get_instance(*args, **kwargs):
        if cls not in instances:
            with instance_lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class MySingleton:
    def __init__(self):
        self.value = None


if __name__ == '__main__':
    s1 = MySingleton()
    s2 = MySingleton()
    print(s1 is s2)
