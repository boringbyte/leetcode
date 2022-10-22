from abs_class import AbstractClass


class NullClass(AbstractClass):
    def do_something(self, value):
        print(f'Not doing {value}')
