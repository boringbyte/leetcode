from abs_builder import AbstractBuilder


class Builder(AbstractBuilder):

    def get_case(self):
        self._computer.case = 'Coolermaster'

    def build_mainboard(self):
        self._computer.mainboard = 'MSI'
        self._computer.cpu = 'Intel Core i3'
        self._computer.memory = 'Corsair Vengeance 8GB'

    def install_mainboard(self):
        pass

    def install_video_card(self):
        self._computer.video_card = 'GeForce GTX 3070'

    def install_hard_drive(self):
        self._computer.hard_drive = 'Seagate 2TB'
