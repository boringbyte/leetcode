import copy


class Address:

    def __init__(self, street, city):
        self.street = street
        self.city = city

    def __str__(self):
        return f"{self.street}, {self.city}"


class Person:

    def __init__(self, name, age, address):
        self.name = name
        self.age = age
        self.address = address

    def deep_clone(self):
        return copy.deepcopy(self)

    def __str__(self):
        return f"{self.name}, {self.age}, lives at {self.address}"


if __name__ == '__main__':

    # Original
    original_address = Address("123 Main St", "Boston")
    original_person = Person("Alice", 30, original_address)

    # Shallow Clone
    cloned_person = original_person.deep_clone()
    cloned_person.name = "Bob"
    cloned_person.age = 25

    # Change nested object
    cloned_person.address.city = "New York"

    print("Original: ", original_person)
    print("Clone: ", cloned_person)
