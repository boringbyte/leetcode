from abs_state import AbstractState


class Empty(AbstractState):

    def add_item(self):
        self._cart.items += 1
        print('You added the first item')
        self._cart.state = self._cart.not_empty

    def remove_item(self):
        print('Your cart is empty! Nothing to remove!!')

    def checkout(self):
        print('Your cart is empty. Go shopping')

    def pay(self):
        print('Your cart is empty. How did you get here?')

    def empty_cart(self):
        print('Your is already empty.')
