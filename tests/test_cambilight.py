import pytest


@pytest.fixture
def mock_lifxlan(mocker):
    return mocker.patch('lifxlan')
