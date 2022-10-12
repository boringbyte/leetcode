class Computer(object):
    case: str
    mainboard: str
    cpu: str
    memory: str
    hard_drive: str
    video_card: str

    def display(self):
        print('Custom Computer:')
        print(f'\t{"Case":>10}: {self.case}')
        print(f'\t{"Mainboard":>10}: {self.mainboard}')
        print(f'\t{"CPU":>10}: {self.cpu}')
        print(f'\t{"Memory":>10}: {self.memory}')
        print(f'\t{"Hard Drive":>10}: {self.hard_drive}')
        print(f'\t{"Video Card":>10}: {self.video_card}')


class MyComputer(object):

    def get_computer(self):
        return self._computer

    def build_computer(self):
        computer = self._computer = Computer()
        computer.case = 'Coolermaster'
        computer.mainboard = 'MSI'
        computer.cpu = 'Intel Core i9'
        computer.memory = '2 X 16GB'
        computer.hard_drive = 'SSD 2TB'
        computer.video_card = 'GeForce'
