import pytest

from marimo_scipy_utils import abbrev_format


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0.0"),
        (42.7, "42.7"),
        (999, "999.0"),
        (1000, "1k"),
        (1500, "2k"),  # round-half-to-even
        (999_000, "999k"),
        (1_000_000, "1M"),
        (2_400_000, "2M"),
        (2_500_000, "2M"),  # round-half-to-even
        (999_000_000, "999M"),
        (1_000_000_000, "1B"),
        (3e9, "3B"),
    ],
)
def test_positive_values(value, expected):
    assert abbrev_format(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1500, "-2k"),
        (-2_000_000, "-2M"),
        (-3e9, "-3B"),
        (-42.7, "-42.7"),
    ],
)
def test_negative_values(value, expected):
    assert abbrev_format(value) == expected


def test_pos_argument_is_ignored():
    assert abbrev_format(1000, 3) == abbrev_format(1000)
