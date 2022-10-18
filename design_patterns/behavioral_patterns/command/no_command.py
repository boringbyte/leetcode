from abs_command import AbstractCommand
from abs_order_command import AbstractOrderCommand


class NoCommand(AbstractCommand):

    def __init__(self, args):
        self._command = args[0]
        pass

    def execute(self):
        print(f'No command named {self._command}')

