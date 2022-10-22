from my_object_factory import MyObjectFactory


if __name__ == '__main__':
    """
        - Provides a default object
        - The default object need do nothing
        - Eliminate tests for None
        - Clients can just use the object returned
        - Not limited to classes
        - Useful for functions, iterator, generator
    """
    my_obj = MyObjectFactory.create_object('myclass')
    my_obj.do_something('something')
