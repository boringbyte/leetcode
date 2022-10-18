from abs_strategy import AbstractStrategy


class UPSStrategy(AbstractStrategy):
    def calculate(self, order):
        return 4.00
