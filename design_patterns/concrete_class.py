from design_patterns.abstract_class import MyABC


class MyClass(MyABC):
    """Implementation of abstract base class"""

    def __init__(self, price=None):
        self._price = price

    def do_something(self, value):
        """Implementation of abstract method"""
        self._price *= value

    @property
    def price(self):
        # https://www.freecodecamp.org/news/python-property-decorator/
        """Implementation of abstract property"""
        return self._price

    @price.setter
    def price(self, new_price):
        if new_price > 0 and isinstance(new_price, float):
            self._price = new_price
        else:
            print('Please enter a valid price')

    @price.deleter
    def price(self):
        del self._price
