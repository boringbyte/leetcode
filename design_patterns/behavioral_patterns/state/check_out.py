from abs_state import AbstractState


class AtCheckOut(AbstractState):

    def add_item(self):
        self._cart.items += 1
        print(f'You cant add items at the check out counter.!')

    def remove_item(self):
        self._cart.items -= 1
        if self._cart.items:
            print(f'You now have {self._cart.items} items in your cart.')
        else:
            print('Your cart is empty! Nothing to remove!!')
            self._cart.state = self._cart.empty

    def checkout(self):
        print('You are already at checkout.')

    def pay(self):
        print(f'You paid for {self._cart.items} items')
        self._cart.state = self._cart.paid_for

    def empty_cart(self):
        self._cart.items = 0
        self._cart.state = self._cart.empty
        print('Your cart is empty again')
