from order import Order
from shipping_cost import ShippingCost
from fedex_strategy import FedExStrategy
from usps_strategy import USPSStrategy
from ups_strategy import UPSStrategy


if __name__ == '__main__':
    """
        1. Encapsulates algorithms
        2. Several techniques available
            - Class per algorithm
            - Function definitions
            - Lambda expressions
        3. Sequences of if/elif/else are a red flag
    """
    order = Order()
    strategy = FedExStrategy()
    cost_calculator = ShippingCost(strategy)
    cost = cost_calculator.shipping_cost(order)
    assert cost == 3.0

    order = Order()
    strategy = UPSStrategy()
    cost_calculator = ShippingCost(strategy)
    cost = cost_calculator.shipping_cost(order)
    assert cost == 4.0

    order = Order()
    strategy = USPSStrategy()
    cost_calculator = ShippingCost(strategy)
    cost = cost_calculator.shipping_cost(order)
    assert cost == 5.0

    print('Tests Passed!')
