import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats

import marimo_scipy_utils.marimo_components as components
from marimo_scipy_utils import (
    DistributionConfigurationError,
    InputVar,
    MissingParameterError,
    ParameterRangeError,
    UnknownDistributionError,
    display_sliders,
    params_sliders,
    resolve_distribution,
    sample_invars,
)
from marimo_scipy_utils.marimo_components import _dist_plot, _parameter_descriptions


def test_params_sliders_configuration():
    sliders = params_sliders(
        {
            "mean": {"lower": 0, "upper": 100, "step": 1, "value": 50},
            "std": {"lower": 0, "upper": 10},
        },
    )
    assert sliders["mean"].start == 0
    assert sliders["mean"].stop == 100
    assert sliders["mean"].step == 1
    assert sliders["mean"].value == 50
    # defaults: step 0.1, value at the midpoint
    assert sliders["std"].step == 0.1
    assert sliders["std"].value == 5


def test_params_sliders_missing_bound_raises():
    with pytest.raises(MissingParameterError):
        params_sliders({"mean": {"lower": 0}})


def test_params_sliders_inverted_bound_raises():
    with pytest.raises(ParameterRangeError):
        params_sliders({"mean": {"lower": 10, "upper": 5}})


def test_resolve_distribution():
    assert resolve_distribution("norm") is scipy.stats.norm
    assert resolve_distribution(scipy.stats.beta) is scipy.stats.beta
    # any scipy.stats distribution works by name, not just the curated ones
    assert resolve_distribution("poisson") is scipy.stats.poisson
    with pytest.raises(UnknownDistributionError):
        resolve_distribution("not_a_dist")


def test_display_sliders_requires_dist_for_multiple_sliders():
    sliders = params_sliders({"loc": {"lower": 0, "upper": 10}})
    with pytest.raises(DistributionConfigurationError):
        display_sliders("x", sliders, {})


def test_display_sliders_unknown_distribution_raises():
    sliders = params_sliders({"loc": {"lower": 0, "upper": 10}})
    with pytest.raises(UnknownDistributionError):
        display_sliders("x", sliders, {}, dist="not_a_dist")


def test_display_sliders_registers_distribution_invar():
    invars = {}
    sliders = params_sliders(
        {"loc": {"lower": 0, "upper": 10}, "scale": {"lower": 0.1, "upper": 5}},
    )
    component = display_sliders("Growth", sliders, invars, dist="norm")
    assert component is not None
    var = invars["Growth"]
    assert isinstance(var, InputVar)
    assert var.dist is scipy.stats.norm
    assert not var.is_constant


def test_display_sliders_registers_constant_invar():
    invars = {}
    slider = mo.ui.slider(start=0, stop=10, value=3)
    component = display_sliders("Rate", slider, invars)
    assert component is not None
    var = invars["Rate"]
    assert var.is_constant
    assert var.params.value == 3


def test_display_sliders_discrete_distribution():
    invars = {}
    sliders = params_sliders({"mu": {"lower": 1, "upper": 20, "value": 5}})
    component = display_sliders("Events", sliders, invars, dist="poisson")
    assert component is not None
    assert invars["Events"].dist is scipy.stats.poisson


def test_dist_plot_closes_figure():
    open_before = plt.get_fignums()
    _dist_plot({"loc": 0, "scale": 1}, scipy.stats.norm)
    assert plt.get_fignums() == open_before


def test_dist_plot_caps_discrete_points(monkeypatch):
    steps = []
    original_arange = np.arange

    def arange_spy(start, stop, step=1):
        steps.append(step)
        return original_arange(start, stop, step)

    class WideDiscreteDistribution:
        def ppf(self, q):
            return 0 if q < 0.5 else 10_000

        def pmf(self, x):
            return np.ones_like(x, dtype=float)

    def wide_discrete_dist():
        return WideDiscreteDistribution()

    monkeypatch.setattr(components.np, "arange", arange_spy)

    _dist_plot({}, wide_discrete_dist)

    assert any(step > 1 for step in steps)


def test_parameter_descriptions_for_callable_distribution():
    sliders = params_sliders(
        {
            "loc": {"lower": 0, "upper": 10},
            "scale": {"lower": 0.1, "upper": 5},
        },
    )

    descriptions = _parameter_descriptions(sliders, scipy.stats.norm, None)

    assert descriptions == {"loc": "Mean", "scale": "Standard deviation"}


def test_input_var_sample_constant():
    var = InputVar(dist=None, params=mo.ui.slider(start=0, stop=10, value=4))
    samples = var.sample(100)
    assert samples.shape == (100,)
    assert (samples == 4).all()


def test_input_var_sample_distribution():
    sliders = params_sliders(
        {
            "loc": {"lower": 0, "upper": 10, "value": 5},
            "scale": {"lower": 0.1, "upper": 5, "value": 1},
        },
    )
    var = InputVar(dist=scipy.stats.norm, params=sliders)
    samples = var.sample(10_000, rng=np.random.default_rng(42))
    assert samples.shape == (10_000,)
    assert abs(samples.mean() - 5) < 0.1


def test_input_var_frozen_raises_for_constant():
    var = InputVar(dist=None, params=mo.ui.slider(start=0, stop=10, value=4))
    with pytest.raises(DistributionConfigurationError):
        var.frozen()


def test_sample_invars():
    invars = {
        "const": InputVar(dist=None, params=mo.ui.slider(start=0, stop=10, value=2)),
        "dist": InputVar(
            dist=scipy.stats.uniform,
            params=params_sliders(
                {
                    "loc": {"lower": 0, "upper": 1, "value": 0},
                    "scale": {"lower": 1, "upper": 2, "value": 1},
                },
            ),
        ),
    }
    samples = sample_invars(invars, 50, rng=np.random.default_rng(0))
    assert set(samples) == {"const", "dist"}
    assert samples["const"].shape == (50,)
    assert (samples["const"] == 2).all()
    assert ((samples["dist"] >= 0) & (samples["dist"] <= 1)).all()


def test_sample_invars_uses_one_generator_for_integer_seed():
    invars = {
        "first": InputVar(
            dist=scipy.stats.uniform,
            params=params_sliders(
                {"loc": {"lower": 0, "upper": 1}, "scale": {"lower": 1, "upper": 2}},
            ),
        ),
        "second": InputVar(
            dist=scipy.stats.uniform,
            params=params_sliders(
                {"loc": {"lower": 0, "upper": 1}, "scale": {"lower": 1, "upper": 2}},
            ),
        ),
    }

    samples = sample_invars(invars, 20, rng=0)

    assert not np.array_equal(samples["first"], samples["second"])
