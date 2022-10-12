from design_patterns.creational_patterns.abstract_factory.autos.abs_auto import AbstractAuto


class LincolnMKS(AbstractAuto):

    def start(self):
        print('Lincoln MKS running smoothly')

    def stop(self):
        print('Lincoln MKS shutting down.')
