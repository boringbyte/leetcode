from mock_vendors import MOCK_VENDORS
CUSTOMERS = MOCK_VENDORS


def main():
    for cust in CUSTOMERS:
        print(f'Name: {cust.name}; Address: {cust.address}')


if __name__ == '__main__':
    """
    Object Adapter:
        1. Composition over inheritance
        2. Delegate to the adaptee
        3. Works with all adaptee subclasses
    Class Adapter:
        1. Subclassing
        2. Override adaptee methods
        3. Committed to one adaptee subclass
    """
    main()
