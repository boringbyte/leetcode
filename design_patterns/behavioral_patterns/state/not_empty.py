from abs_state import AbstractState


class NotEmpty(AbstractState):

    def add_item(self):
        self._cart.items += 1
        print(f'You now have {self._cart.items} items in your cart.')

    def remove_item(self):
        self._cart.items -= 1
        if self._cart.items:
            print(f'You now have {self._cart.items} items in your cart.')
        else:
            print('Your cart is empty! Nothing to remove!!')
            self._cart.state = self._cart.empty

    def checkout(self):
        print('Done shopping Lets check out !')
        self._cart.state = self._cart.check_out

    def pay(self):
        print('You have to go to check out to pay!')

    def empty_cart(self):
        print('You can only empty the cart at checkout.')
