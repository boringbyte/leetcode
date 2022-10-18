from shopping_cart import ShoppingCart


def main():
    print('====> first cart')
    cart = ShoppingCart()
    cart.add_item()
    cart.remove_item()
    cart.add_item()
    cart.add_item()
    cart.add_item()
    cart.remove_item()
    cart.checkout()
    cart.pay()

    print('====> second cart')
    cart = ShoppingCart()
    cart.add_item()
    cart.add_item()
    cart.checkout()
    cart.empty_cart()
    cart.add_item()
    cart.checkout()
    cart.pay()

    print('====> Expect an error here')
    cart.add_item()


if __name__ == '__main__':
    """
    1. Encapsulates state-specific behavior
    2. Distributes behavior across state classes
    3. Makes sate transitions explicit
    4. State objects can be shared
    5. Flexible transition definitions
    6. Can create states at transition time.
    """
    main()
