from iheart24 import IHeart42
import random


def main():
    g = IHeart42('Arthur')

    print(f'Hero: {g.game_state.name}, Game level: {g.game_state.level}')
    memento = g.create_memento()

    g.game_state.level = random.randint(1, 42)
    g.game_state.name = 'Ford'
    print(f'Hero: {g.game_state.name}, Game level: {g.game_state.level}')

    g.set_memento(memento)
    print(f'Hero: {g.game_state.name}, Game level: {g.game_state.level}')


if __name__ == '__main__':
    """
        - Preserves encapsulation
        - Simplifies the Originator class.
        - Easy to implement state restoration.
        - Using mementos might be costly.
        - Caretaker class can be memory intensive.
        - Python introspection can break encapsulation
    """
    main()
