from abs_command import AbstractCommand
from abs_order_command import AbstractOrderCommand


class ShipOrder(AbstractCommand, AbstractOrderCommand):
    name = description = 'ShipOrder'

    def execute(self):
        raise NotImplementedError
