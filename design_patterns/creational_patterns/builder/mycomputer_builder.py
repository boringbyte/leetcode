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


class MyComputerBuilder(object):

    def get_computer(self):
        return self._computer

    def build_computer(self):
        self._computer = Computer()
        self.get_case()
        self.build_mainboard()
        self.install_hard_drive()
        self.install_video_card()

    def get_case(self):
        self._computer.case = 'Coolermaster N300'

    def install_mainboard(self):
        pass

    def build_mainboard(self):
        self._computer.mainboard = 'MSI 970'
        self._computer.cpu = 'Intel Core i7-4770'
        self._computer.memory = 'Corsair Vengeance 16GB'

    def install_video_card(self):
        self._computer.video_card = 'GeForce GTX 3070'

    def install_hard_drive(self):
        self._computer.hard_drive = 'Seagate 4TB'
