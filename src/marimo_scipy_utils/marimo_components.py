"""Utility functions for creating interactive marimo UI with scipy distributions.

This module provides functions for creating and configuring interactive UI elements
in marimo notebooks, with a focus on parameter input and visualization using scipy
probability distributions.

The module includes utilities for:
- Creating interactive sliders with distribution plots
- Generating and validating parameter ranges for distributions
- Formatting numbers with k/M/B suffixes
- Creating dictionaries of parameter sliders
- Sampling configured input variables for Monte Carlo simulation
"""

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.ticker import FuncFormatter

from .exceptions import (
    DistributionConfigurationError,
    MissingParameterError,
    ParameterBoundError,
    UnknownDistributionError,
)

_DEFAULT_STEP = 0.1
_DEFAULT_FIGSIZE = (2.0, 2.0)
_DEFAULT_COLOR = "C0"
_DEFAULT_TAIL = 0.0005
_PLOT_POINTS = 100


class RangeSpec(TypedDict, total=False):
    """Range specification for a single distribution parameter.

    Keys:
        description: Human-readable label for the parameter.
        lower: Lower bound for the parameter (None means it must be provided).
        upper: Upper bound for the parameter (None means it must be provided).
        step: Slider step size (defaults to 0.1).
        value: Initial slider value (defaults to the midpoint of the bounds).
    """

    description: str
    lower: float | None
    upper: float | None
    step: float
    value: float


# Curated parameter metadata for commonly-used distributions. Other scipy.stats
# distributions are still usable by name in `display_sliders`; they just lack
# curated descriptions and bounds, so `generate_ranges` requires one of these.
_distributions: dict[str, dict[str, RangeSpec]] = {
    "triang": {
        "c": {"description": "Center (% of width)", "lower": 0, "upper": 1},
        "loc": {"description": "Lower bound", "lower": None, "upper": None},
        "scale": {"description": "Width", "lower": 0, "upper": None},
    },
    "norm": {
        "loc": {"description": "Mean", "lower": None, "upper": None},
        "scale": {"description": "Standard deviation", "lower": 0, "upper": None},
    },
    "uniform": {
        "loc": {"description": "Lower bound", "lower": None, "upper": None},
        "scale": {"description": "Width", "lower": 0, "upper": None},
    },
    "beta": {
        "a": {"description": "Alpha (a > 0)", "lower": 0, "upper": None},
        "b": {"description": "Beta (b > 0)", "lower": 0, "upper": None},
        "loc": {"description": "Lower bound", "lower": None, "upper": None},
        "scale": {"description": "Width", "lower": 0, "upper": None},
    },
}

# Mapping of curated distribution names to scipy callables.
SCIPY_DISTRIBUTIONS: dict[str, Callable] = {
    name: getattr(scipy.stats, name) for name in _distributions
}


def resolve_distribution(dist: str | Callable) -> Callable:
    """Resolve a distribution name or callable to a scipy distribution.

    Args:
        dist: Either a scipy.stats distribution object or the name of one
            (e.g. "norm", "poisson").

    Returns:
        The scipy distribution callable.

    Raises:
        UnknownDistributionError: If the name does not match a scipy.stats
            distribution.

    """
    if callable(dist):
        return dist
    name = str(dist)
    resolved = getattr(scipy.stats, name, None)
    if resolved is None or not callable(resolved):
        raise UnknownDistributionError(name)
    return resolved


@dataclass
class InputVar:
    """A configured input variable: either a constant or a distribution.

    Attributes:
        dist: The scipy distribution callable, or None for a constant.
        params: The slider(s) controlling the variable. A single slider for a
            constant, or a dictionary of sliders for distribution parameters.

    """

    dist: Callable | None
    params: mo.ui.dictionary | mo.ui.slider

    @property
    def is_constant(self) -> bool:
        """Whether this variable is a constant rather than a distribution."""
        return self.dist is None

    def frozen(self) -> object:
        """Return the frozen scipy distribution for the current slider values.

        Raises:
            DistributionConfigurationError: If this variable is a constant.

        """
        if self.dist is None:
            raise DistributionConfigurationError
        return self.dist(**{k: v.value for k, v in self.params.items()})

    def sample(
        self,
        size: int,
        rng: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        """Draw samples from the variable.

        Constants return an array filled with the current slider value.

        Args:
            size: Number of samples to draw.
            rng: Optional numpy random Generator or seed for reproducibility.

        Returns:
            np.ndarray: Array of `size` samples.

        """
        if self.dist is None:
            return np.full(size, self.params.value)
        return self.frozen().rvs(size=size, random_state=rng)


def sample_invars(
    invars: dict[str, InputVar],
    size: int,
    rng: np.random.Generator | int | None = None,
) -> dict[str, np.ndarray]:
    """Sample every input variable, e.g. for a Monte Carlo simulation.

    Args:
        invars: Dictionary of input variables built up by `display_sliders`.
        size: Number of samples to draw per variable.
        rng: Optional numpy random Generator or seed for reproducibility.

    Returns:
        dict[str, np.ndarray]: Mapping of variable name to sample array.

    """
    return {name: var.sample(size, rng) for name, var in invars.items()}


def _deep_merge(dict1: dict, dict2: dict) -> dict:
    """Recursively merge two dicts without mutating either input."""
    result = dict(dict1)
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def generate_ranges(
    distribution: str,
    ranged_distkwargs: dict[str, RangeSpec],
) -> dict[str, RangeSpec]:
    """Generate parameter ranges for a probability distribution.

    Takes a distribution name and dictionary of parameter ranges, validates the ranges
    against allowed bounds for that distribution, and returns a merged dictionary with
    complete range specifications.

    Args:
        distribution: Name of the probability distribution (e.g. "norm", "beta").
            Must be one of the curated distributions with known parameter bounds.
        ranged_distkwargs: Dictionary mapping parameter names to their range
            specifications. Each range spec should have "lower" and "upper" bounds.

    Returns:
        dict[str, RangeSpec]: Complete parameter range specifications with
            distribution defaults merged with provided ranges.

    Raises:
        UnknownDistributionError: If the distribution has no curated parameter bounds
        MissingParameterError: If a required parameter range is not provided
        ParameterBoundError: If a provided range exceeds the allowed bounds for a param

    """
    if distribution not in _distributions:
        raise UnknownDistributionError(distribution, sorted(_distributions))
    defaults = copy.deepcopy(_distributions[distribution])

    for p_name, ranges in defaults.items():
        if p_name not in ranged_distkwargs:
            if ranges["lower"] is None or ranges["upper"] is None:
                raise MissingParameterError(p_name, ranges)
            continue

        given = ranged_distkwargs[p_name]

        if "lower" not in given and ranges["lower"] is None:
            raise MissingParameterError(p_name)
        if (
            "lower" in given
            and ranges["lower"] is not None
            and given["lower"] < ranges["lower"]
        ):
            raise ParameterBoundError(
                p_name,
                given["lower"],
                ranges["lower"],
                "lower",
            )

        if "upper" not in given and ranges["upper"] is None:
            raise MissingParameterError(p_name)
        if (
            "upper" in given
            and ranges["upper"] is not None
            and given["upper"] > ranges["upper"]
        ):
            raise ParameterBoundError(
                p_name,
                given["upper"],
                ranges["upper"],
                "upper",
            )

    return _deep_merge(defaults, ranged_distkwargs)


def params_sliders(
    ranged_distkwargs: dict[str, RangeSpec],
) -> mo.ui.dictionary:
    """Create a dictionary of sliders for parameter ranges.

    Takes a dictionary of param ranges and creates interactive sliders for each param.
    The sliders will be bounded by the lower/upper values specified in the ranges dict.
    The step size and initial value can optionally be specified per param.

    Args:
        ranged_distkwargs: Dictionary mapping parameter names to their range
            specifications. Each range spec should have "lower" and "upper" bounds,
            and optionally "step" and "value" keys.

    Returns:
        mo.ui.dictionary: A dictionary of marimo slider UI elements, one per parameter.
            Each slider will be configured according to the parameter's range spec.

    Example:
        >>> ranges = {
        ...     "mean": {"lower": 0, "upper": 100, "step": 1, "value": 50},
        ...     "std": {"lower": 0, "upper": 10}
        ... }
        >>> sliders = params_sliders(ranges)

    """
    return mo.ui.dictionary(
        {
            p_name: mo.ui.slider(
                start=ranges["lower"],
                stop=ranges["upper"],
                step=ranges.get("step", _DEFAULT_STEP),
                value=ranges.get("value", (ranges["lower"] + ranges["upper"]) / 2),
            )
            for p_name, ranges in ranged_distkwargs.items()
        },
    )


_thousand = 1e3
_million = 1e6
_billion = 1e9


def abbrev_format(x: float, pos: int | None = None) -> str:  # noqa: ARG001
    """Format numbers with k/M/B suffixes for thousands/millions/billions.

    Uses Python's default round-half-to-even, so e.g. 2500000 formats as "2M".
    Negative numbers keep their sign: -1500 formats as "-2k".

    Args:
        x: The number to format
        pos: The tick position (unused but required by matplotlib formatter interface)

    Returns:
        str: The formatted number string with k/M/B suffix if applicable

    Example:
        >>> abbrev_format(1500)
        '2k'
        >>> abbrev_format(2400000)
        '2M'
        >>> abbrev_format(3e9)
        '3B'
        >>> abbrev_format(-1500)
        '-2k'
        >>> abbrev_format(42.7)
        '42.7'

    """
    magnitude = abs(x)
    if magnitude >= _billion:
        return f"{x / _billion:.0f}B"
    if magnitude >= _million:
        return f"{x / _million:.0f}M"
    if magnitude >= _thousand:
        return f"{x / _thousand:.0f}k"
    return f"{x:.1f}"


def _dist_plot(
    params: dict,
    dist: Callable,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    color: str = _DEFAULT_COLOR,
    tail: float = _DEFAULT_TAIL,
) -> mo.Html:
    frozen = dist(**params)
    x_min = frozen.ppf(tail)
    x_max = frozen.ppf(1 - tail)
    is_continuous = hasattr(frozen, "pdf")

    fig, ax = plt.subplots(figsize=figsize)
    try:
        if is_continuous:
            x = np.linspace(x_min, x_max, _PLOT_POINTS)
            y = frozen.pdf(x)
            ax.plot(x, y, color=color, linewidth=2)
            ax.fill_between(x, y, color=color, alpha=0.3)
        else:
            x = np.arange(math.floor(x_min), math.ceil(x_max) + 1)
            y = frozen.pmf(x)
            ax.vlines(x, 0, y, color=color, linewidth=2)
        ax.set_ylabel("Probability Density" if is_continuous else "Probability Mass")
        ax.grid(visible=True, alpha=0.3)
        ax.tick_params(axis="y", labelleft=False)
        ax.xaxis.set_major_formatter(FuncFormatter(abbrev_format))
        return mo.as_html(fig)
    finally:
        plt.close(fig)


def _parameter_descriptions(
    sliders: mo.ui.dictionary,
    dist: str | Callable,
    descriptions: dict[str, str] | None,
) -> dict[str, str]:
    if descriptions:
        return descriptions
    if isinstance(dist, str) and dist in _distributions:
        spec = _distributions[dist]
        return {k: spec[k].get("description", k) for k in sliders if k in spec}
    return {}


def _display_sliders_with_plot(  # noqa: PLR0913
    name: str,
    sliders: mo.ui.dictionary,
    dist: str | Callable,
    invars: dict[str, InputVar],
    descriptions: dict[str, str] | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    color: str = _DEFAULT_COLOR,
    tail: float = _DEFAULT_TAIL,
) -> mo.Html:
    _dist = resolve_distribution(dist)
    parameter_descriptions = _parameter_descriptions(sliders, dist, descriptions)
    html = mo.Html(
        "<table>"
        + "\n".join(
            [
                f"<tr><td>{parameter_descriptions.get(k, k)}</td><td>{v}</td></tr>"
                for k, v in sliders.items()
            ],
        )
        + "</table>",
    )
    invars[name] = InputVar(dist=_dist, params=sliders)
    return mo.hstack(
        [
            mo.vstack([mo.md(f"### {name}"), html]),
            _dist_plot(
                {k: v.value for k, v in sliders.items()},
                _dist,
                figsize=figsize,
                color=color,
                tail=tail,
            ),
        ],
        align="start",
        widths=[2, 1],
    )


def display_sliders(  # noqa: PLR0913
    name: str,
    sliders: mo.ui.dictionary | mo.ui.slider,
    invars: dict[str, InputVar],
    dist: str | Callable | None = None,
    descriptions: dict[str, str] | None = None,
    *,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    color: str = _DEFAULT_COLOR,
    tail: float = _DEFAULT_TAIL,
) -> mo.Html:
    """Display parameter sliders with optional distribution plot.

    Args:
        name: Name of the parameter group to display
        sliders: Either a single slider or dictionary of sliders for distribution params
        invars: Dictionary in which the configured `InputVar` is stored under `name`
        dist: Distribution to use (string name or callable), required for multi-sliders
        descriptions: Optional dict mapping parameter names to descriptions
        figsize: Size of the distribution plot in inches
        color: Matplotlib color for the distribution plot
        tail: Probability mass excluded from each end of the plotted range

    Returns:
        Marimo component displaying the sliders and optional distribution plot

    For a single slider, displays it as a constant parameter.
    For multiple sliders, displays them with a plot of the resulting distribution.
    Distribution must be specified for multiple sliders, either as a string name
    matching a scipy distribution or as a callable distribution object.

    Raises:
        DistributionConfigurationError: If dist is None for multiple sliders
        UnknownDistributionError: If dist names an unknown scipy distribution

    """
    if isinstance(sliders, mo.ui.dictionary):
        if dist is None:
            raise DistributionConfigurationError
        return _display_sliders_with_plot(
            name,
            sliders,
            dist,
            invars,
            descriptions,
            figsize=figsize,
            color=color,
            tail=tail,
        )
    # Single slider is a constant
    invars[name] = InputVar(dist=None, params=sliders)
    return mo.vstack(
        [mo.md(f"### {name} = {abbrev_format(sliders.value)}"), sliders],
    )
