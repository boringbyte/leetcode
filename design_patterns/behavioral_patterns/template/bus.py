from abs_transport import AbstractTransport


class Bus(AbstractTransport):

    def start_engine(self):
        print('Starting the Cummins diesel engines')

    def travel_to_destination(self):
        print('Driving')
