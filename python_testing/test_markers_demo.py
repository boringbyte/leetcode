import pytest
import sys

@pytest.mark.smoke
def test_login():
    print('Login done')


@pytest.mark.regression
def test_add_product():
    print('Added product done')


@pytest.mark.smoke
def test_logout():
    print('Logout done')


@pytest.mark.skipif(sys.version_info.minor < 10, reason='Current support version is >= 10')
def test_version_info():
    print('Current version of the python')


@pytest.mark.xfail
def test_fail():
    assert False
    print('Assertion failed')


def test_pass():
    assert True
    print('Assertion passed')
