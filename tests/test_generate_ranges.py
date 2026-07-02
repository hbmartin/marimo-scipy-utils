import pytest

from marimo_scipy_utils import (
    MissingParameterError,
    ParameterBoundError,
    UnknownDistributionError,
    generate_ranges,
)
from marimo_scipy_utils.marimo_components import _deep_merge, _distributions


def test_merges_defaults_with_given_ranges():
    result = generate_ranges(
        "norm",
        {
            "loc": {"lower": 0, "upper": 100},
            "scale": {"upper": 10},
        },
    )
    assert result["loc"]["lower"] == 0
    assert result["loc"]["upper"] == 100
    assert result["loc"]["description"] == "Mean"
    # scale keeps the curated lower bound of 0
    assert result["scale"]["lower"] == 0
    assert result["scale"]["upper"] == 10


def test_zero_lower_bound_is_not_treated_as_missing():
    """A curated lower bound of 0 must satisfy the bound requirement (0 is falsy)."""
    result = generate_ranges(
        "triang",
        {
            "loc": {"lower": 0, "upper": 100},
            "scale": {"upper": 50},
        },
    )
    # "c" has concrete bounds (0, 1) and so may be omitted entirely
    assert result["c"]["lower"] == 0
    assert result["c"]["upper"] == 1
    assert result["scale"]["lower"] == 0


def test_omitting_param_with_unbounded_side_raises():
    with pytest.raises(MissingParameterError):
        generate_ranges("norm", {"scale": {"upper": 10}})  # loc omitted


def test_provided_param_missing_required_lower_raises():
    with pytest.raises(MissingParameterError):
        generate_ranges(
            "norm",
            {"loc": {"upper": 100}, "scale": {"upper": 10}},
        )


def test_provided_param_missing_required_upper_raises():
    with pytest.raises(MissingParameterError):
        generate_ranges(
            "norm",
            {"loc": {"lower": 0, "upper": 100}, "scale": {"lower": 1}},
        )


def test_lower_bound_violation_raises():
    with pytest.raises(ParameterBoundError):
        generate_ranges(
            "norm",
            {"loc": {"lower": 0, "upper": 100}, "scale": {"lower": -1, "upper": 10}},
        )


def test_upper_bound_violation_raises():
    with pytest.raises(ParameterBoundError):
        generate_ranges(
            "triang",
            {
                "c": {"lower": 0, "upper": 2},
                "loc": {"lower": 0, "upper": 100},
                "scale": {"upper": 50},
            },
        )


def test_unknown_distribution_raises():
    with pytest.raises(UnknownDistributionError):
        generate_ranges("not_a_dist", {})


def test_does_not_mutate_module_defaults():
    before = repr(_distributions)
    generate_ranges(
        "norm",
        {"loc": {"lower": 0, "upper": 100}, "scale": {"upper": 10}},
    )
    assert repr(_distributions) == before


def test_deep_merge_does_not_mutate_inputs():
    d1 = {"a": {"x": 1, "y": 2}, "b": 3}
    d2 = {"a": {"y": 20, "z": 30}}
    merged = _deep_merge(d1, d2)
    assert merged == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}
    assert d1 == {"a": {"x": 1, "y": 2}, "b": 3}
    assert d2 == {"a": {"y": 20, "z": 30}}
