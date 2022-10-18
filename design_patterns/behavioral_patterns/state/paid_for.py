from abs_state import AbstractState


class PaidFor(AbstractState):

    def add_item(self):
        print('You already paid for your purchases. Want to shop some more? Get a new shopping cart!')

    def remove_item(self):
        print('You have already paid for your purchases and cannot remove any.')

    def checkout(self):
        print('Why are you back here? You already paid!')

    def pay(self):
        print('You already paid. You cannot pay twice!')

    def empty_cart(self):
        print('You paid already. Time to go home!')
