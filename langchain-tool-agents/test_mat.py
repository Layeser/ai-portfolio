from mat import *
import pytest 

@pytest.fixture
def samplelist():
    return [1, 2, 3, 4, 5]

def test_sum(samplelist):
    assert sum(samplelist) == 15

def test_max(samplelist):
    assert max(samplelist) == 5

@pytest.mark.parametrize("input_list, expected", [
    ([1, 2, 3], 6),
    ([0, 0, 0], 0),
    ([-1, -2, -3], -6)
])
def test_sum_param(input_list, expected):
    assert sum(input_list) == expected