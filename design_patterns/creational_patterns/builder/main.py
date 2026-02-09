from computer import Computer
from my_computer import MyComputer
from mycomputer_builder import MyComputerBuilder
from builder import Builder
from director import Director

computer = Computer(case='CoolerMaster',
                    mainboard='MSI',
                    cpu='Intel Core i9',
                    memory='2 X 16GB',
                    hard_drive='SSD 2TB',
                    video_card='GeForce')


if __name__ == '__main__':
    """
    1. Long parameter list is a sign that we can do better than this
    2. This breaks the open closed principle as we don't know which one are mandatory and which one aren't
    3. Initializer might grow over a period of time.
    """
    computer.display()
    print("\n")

    """
    This is 2nd attempt
    1. Reduced number of parameters
    2. We are not Exposing attributes to client anymore
    3. We have Encapsulated the attributes of the main class from the client by putting it in MyCompute class. 
       It acts as a builder
    4. But still there is problem of order. we cannot build in any order
    """
    builder = MyComputer()
    builder.build_computer()
    computer = builder.get_computer()
    computer.display()
    print("\n")

    """
    This is 3rd update
    """
    builder = MyComputerBuilder()
    builder.build_computer()
    computer = builder.get_computer()
    computer.display()
    print("\n")

    """
    This is 4th attempt
    1. There is abstract builder
    2. There is concrete builder
    3. There is a director who takes in a concrete builder
    This pattern separates the "how" from the "what"
    Assembly separate from components
    Encapsulates what varies
    Permits different representations
    Client creates Director object
    Director uses concrete builder
    Builder adds parts to the product
    Client receives the product from the builder
    """
    print('This is attempt 4')
    computer_builder = Director(Builder())
    computer_builder.build_computer()
    computer = computer_builder.get_computer()
    computer.display()
