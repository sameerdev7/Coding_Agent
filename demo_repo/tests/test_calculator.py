import pytest

from calculator import average, divide


def test_divide_by_zero_raises():
    with pytest.raises(ValueError):
        divide(10, 0)


def test_average_empty_raises():
    with pytest.raises(ValueError):
        average([])
