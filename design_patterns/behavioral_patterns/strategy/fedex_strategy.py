from abs_strategy import AbstractStrategy


class FedExStrategy(AbstractStrategy):
    def calculate(self, order):
        return 3.00
