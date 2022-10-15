from design_patterns.structural_patterns.adapter.abs_adapter import AbstractAdapter


class VendorAdapter(AbstractAdapter):

    @property
    def name(self):
        return self.adaptee.name

    @property
    def address(self):
        return f'{self.adaptee.number} {self.adaptee.street}'
