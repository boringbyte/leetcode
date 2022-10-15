from mock_vendors import MOCK_VENDORS
CUSTOMERS = MOCK_VENDORS


def main():
    for cust in CUSTOMERS:
        print(f'Name: {cust.name}; Address: {cust.address}')


if __name__ == '__main__':
    main()
