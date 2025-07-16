from collections.abc import Callable

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.stats import beta, norm, triang, uniform

from .exceptions import (
    DistributionConfigurationError,
    MissingParameterError,
    ParameterBoundError,
)

# Dictionary mapping distribution keys to scipy callables
SCIPY_DISTRIBUTIONS: dict[str, Callable] = {
    "uniform": uniform,
    "triang": triang,
    "beta": beta,
    "norm": norm,
}
_CONST = "_constant"
_DEFAULT_STEP = 0.1

_distributions = {
    "triang": {
        "c": {"description": "Center (% of width)", "lower": 0, "upper": 1},
        "loc": {
            "description": "Lower bound",
            "lower": None,
            "upper": None,
        },
        "scale": {
            "description": "Width",
            "lower": 0,
            "upper": None,
        },
    },
    "norm": {
        "loc": {
            "description": "Mean",
            "lower": None,
            "upper": None,
        },
        "scale": {
            "description": "Standard deviation",
            "lower": 0,
            "upper": None,
        },
    },
    "uniform": {
        "loc": {
            "description": "Lower bound",
            "lower": None,
            "upper": None,
        },
        "scale": {
            "description": "Width ",
            "lower": 0,
            "upper": None,
        },
    },
    "beta": {
        "a": {
            "description": "Alpha (a > 0)",
            "lower": 0,
            "upper": None,
        },
        "b": {
            "description": "Beta (b > 0)",
            "lower": 0,
            "upper": None,
        },
        "loc": {
            "description": "Lower bound",
            "lower": None,
            "upper": None,
        },
        "scale": {
            "description": "Width",
            "lower": 0,
            "upper": None,
        },
    },
}


def _deep_merge(result: dict, dict2: dict) -> dict:
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def generate_ranges(distribution: str, ranged_distkwargs: dict) -> dict:
    """Generate parameter ranges for a probability distribution.

    Takes a distribution name and dictionary of parameter ranges, validates the ranges
    against allowed bounds for that distribution, and returns a merged dictionary with
    complete range specifications.

    Args:
        distribution (str): Name of the probability distribution (e.g. "normal", "beta")
        ranged_distkwargs (dict): Dictionary mapping parameter names to their range
            specifications. Each range spec should have "lower" and "upper" bounds.

    Returns:
        dict: Complete parameter range specifications with distribution defaults merged
            with provided ranges

    Raises:
        MissingParameterError: If a required parameter range is not provided
        ParameterBoundError: If a provided range exceeds the allowed bounds for a param

    """
    ranged_copy = _distributions[distribution].copy()

    for p_name, ranges in ranged_copy.items():
        if (
            p_name not in ranged_distkwargs
            and not ranges["lower"]
            and not ranges["upper"]
        ):
            raise MissingParameterError(p_name, ranges)
        if p_name not in ranged_distkwargs:
            continue

        if (
            "lower" in ranged_distkwargs[p_name]
            and ranges["lower"] is not None
            and ranged_distkwargs[p_name]["lower"] < ranges["lower"]
        ):
            raise ParameterBoundError(
                p_name,
                ranged_distkwargs[p_name]["lower"],
                ranges["lower"],
                "lower",
            )
        if "lower" not in ranged_distkwargs[p_name] and ranges["lower"] is None:
            raise MissingParameterError(p_name)

        if (
            "upper" in ranged_distkwargs[p_name]
            and ranges["upper"] is not None
            and ranged_distkwargs[p_name]["upper"] > ranges["upper"]
        ):
            raise ParameterBoundError(
                p_name,
                ranged_distkwargs[p_name]["upper"],
                ranges["upper"],
                "upper",
            )
        if "upper" not in ranged_distkwargs[p_name] and ranges["upper"] is None:
            raise MissingParameterError(p_name)

    return _deep_merge(ranged_copy, ranged_distkwargs)


def params_sliders(
    ranged_distkwargs: dict,
) -> mo.ui.dictionary:
    """Create a dictionary of sliders for parameter ranges.

    Takes a dictionary of param ranges and creates interactive sliders for each param.
    The sliders will be bounded by the lower/upper values specified in the ranges dict.
    The step size and initial value can optionally be specified per param.

    Args:
        ranged_distkwargs (dict): Dictionary mapping parameter names to their range
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
                value=(
                    (ranges["lower"] + ranges["upper"]) / 2
                    if "value" not in ranges
                    else ranges["value"]
                ),
            )
            for p_name, ranges in ranged_distkwargs.items()
        },
    )


_thousand = 1e3
_million = 1e6


def abbrev_format(x: float, pos: int | None) -> str:  # noqa: ARG001
    """Format numbers with k/M suffixes for thousands/millions.

    Args:
        x: The number to format
        pos: The tick position (unused but required by matplotlib formatter interface)

    Returns:
        str: The formatted number string with k/M suffix if applicable

    """
    if x >= _million:
        return f"{x/_million:.0f}M"
    if x >= _thousand:
        return f"{x/_thousand:.0f}k"
    return f"{x:.1f}"


def _dist_plot(params: dict, dist: Callable) -> mo.Html:
    _dist = dist(**params)
    x_min = _dist.ppf(0.0005)
    x_max = _dist.ppf(0.9995)
    x = np.linspace(x_min, x_max, 100)
    pdf_values = _dist.pdf(x)

    fig = plt.figure(figsize=(2, 2))
    plt.plot(x, pdf_values, "b-", linewidth=2, label=None)
    plt.fill_between(x, pdf_values, alpha=0.3)
    plt.ylabel("Probability Density")
    plt.grid(visible=True, alpha=0.3)
    plt.tick_params(axis="y", labelleft=False)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(abbrev_format))
    return mo.as_html(fig)


def _display_sliders_with_plot(
    name: str,
    sliders: mo.ui.dictionary,
    dist: str | Callable,
    invars: dict[str, dict],
    descriptions: dict = {},  # noqa: B006
) -> mo.Html:
    _dist = dist if callable(dist) else SCIPY_DISTRIBUTIONS[str(dist)]
    parameter_descriptions = (
        descriptions
        if descriptions or not isinstance(dist, str)
        else {
            k: (
                k
                if dist not in _distributions
                else _distributions[str(dist)][k].get("description")
            )
            for k, _ in sliders.items()
        }
    )
    html = mo.Html(
        "<table>"
        + "\n".join(
            [
                f"<tr><td>{parameter_descriptions[k]}</td><td>{v}</td></tr>"
                for k, v in sliders.items()
            ],
        )
        + "</table>",
    )
    invars[name] = {"dist": _dist, "params": sliders}
    return mo.hstack(
        [
            mo.vstack([mo.md(f"### {name}"), html]),
            _dist_plot({k: v.value for k, v in sliders.items()}, _dist),
        ],
        align="start",
        widths=[2, 1],
    )


def display_sliders(
    name: str,
    sliders: mo.ui.dictionary | mo.ui.slider,
    invars: dict[str, dict],
    dist: str | Callable | None = None,
    descriptions: dict = {},  # noqa: B006
) -> mo.Html:
    """Display parameter sliders with optional distribution plot.

    Args:
        name: Name of the parameter group to display
        sliders: Either a single slider or dictionary of sliders for distribution params
        invars: Dictionary to store input variable configurations
        dist: Distribution to use (string name or callable), required for multi-sliders
        descriptions: Optional dict mapping parameter names to descriptions

    Returns:
        Marimo component displaying the sliders and optional distribution plot

    For a single slider, displays it as a constant parameter.
    For multiple sliders, displays them with a plot of the resulting distribution.
    Distribution must be specified for multiple sliders, either as a string name
    matching a scipy distribution or as a callable distribution object.

    Raises:
        DistributionConfigurationError: If dist is None for multiple sliders

    """
    if isinstance(sliders, mo.ui.dictionary):
        if dist is None:
            raise DistributionConfigurationError
        return _display_sliders_with_plot(name, sliders, dist, invars, descriptions)
    # Single slider is a constant
    invars[name] = {"dist": _CONST, "params": sliders}
    return mo.vstack(
        [mo.md(f"### {name} = {abbrev_format(sliders.value, None)}"), sliders],
    )
