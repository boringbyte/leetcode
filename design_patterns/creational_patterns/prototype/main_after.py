from design_patterns.creational_patterns.prototype.tower import Tower, MainBoard
from design_patterns.creational_patterns.prototype.laptop import Laptop
from design_patterns.creational_patterns.prototype.prototype_manager import PrototypeManager

if __name__ == '__main__':
    """
    1. Use shallow copy when deep copy isn't required as this is will be easy to compute.
    2. Use deep copy when nested objects are present and all those are required to be copied.
    3. Prototype manager can be used to manage multiple prototypes
    """
    manager = PrototypeManager()

    l1 = Laptop('L1', 'Intel', '32GB', '2TB SSD', 'onboard', '1920X1080')
    manager |= {'L1': l1}
    l1.display()
    l2 = l1.clone()  # manager['L1'].clone()
    l2.model = 'L2'
    l2.processor = 'AMD'
    l2.display()

    t1 = Tower('T1', MainBoard('ASUS', 'Game'), 'AMD', '32 GB', '2TB SSD', 'onboard', '1920X1080')
    t1.display()
    manager |= {'T1': t1}
    t2 = t1.clone()  # manager['T1'].clone()
    t2.model = 'T2'
    t2.mainboard.model = 'Business'
    t1.display()
    t2.display()
