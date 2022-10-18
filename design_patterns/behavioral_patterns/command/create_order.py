from abs_command import AbstractCommand
from abs_order_command import AbstractOrderCommand


class CreateOrder(AbstractCommand, AbstractOrderCommand):

    name = description = 'CreateOrder'

    def execute(self):
        raise NotImplementedError
