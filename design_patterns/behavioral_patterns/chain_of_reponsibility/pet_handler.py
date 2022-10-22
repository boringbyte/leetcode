from abs_handler import AbstractHandler


class PetHandler(AbstractHandler):

    def __init__(self, successor):
        self._successor = successor

    @property
    def successor(self):
        return self._successor

    @successor.setter
    def successor(self, successor):
        self._successor = successor

    def handle(self, request):
        if self.successor:
            self.successor.handle(request)
