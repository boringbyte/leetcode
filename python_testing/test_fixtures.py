import pytest


@pytest.fixture
def setup():
    print('Start Browser')
    yield
    print('Close Browser')


def test_1(setup):
    print('Test 1 executed')


def test_2(setup):
    print('Test 2 executed')


def test_3(setup):
    print('Test 3 executed')
