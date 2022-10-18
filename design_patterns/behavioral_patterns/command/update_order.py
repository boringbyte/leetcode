from abs_command import AbstractCommand
from abs_order_command import AbstractOrderCommand


class UpdateOrder(AbstractCommand, AbstractOrderCommand):
    name = 'UpdateQuantity'
    description = 'UpdateQuantity number'

    def __init__(self, args):
        self.new_qty = args[1]

    def execute(self):
        old_qty = 5
        # Simulate database update
        print('Updated Database')

        # Simulate logging the update
        print(f'Logging: Updated quantity from {old_qty} to {self.new_qty}')
