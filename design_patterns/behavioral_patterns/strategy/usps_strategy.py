from abs_strategy import AbstractStrategy


class USPSStrategy(AbstractStrategy):
    def calculate(self, order):
        return 5.00
