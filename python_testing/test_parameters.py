import pytest


@pytest.mark.parametrize("username, password",
                         [
                             ('lucy', '39090348sdf'),
                             ('lufy', 'sdjfsj9lj')
                         ])
def test_login(username, password):
    print(f'{username}: {password}')
