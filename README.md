[![PyPI](https://img.shields.io/pypi/v/marimo-scipy-utils.svg)](https://pypi.org/project/marimo-scipy-utils/)
[![Lint and Typecheck](https://github.com/hbmartin/marimo-scipy-utils/actions/workflows/lint.yml/badge.svg)](https://github.com/hbmartin/marimo-scipy-utils/actions/workflows/lint.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/🐧️-black-000000.svg)](https://github.com/psf/black)

# Marimo SciPy Utils

Utility functions for creating interactive marimo components with scipy distributions.

This package provides functions for creating and configuring interactive UI elements in marimo notebooks, with a focus on parameter input and visualization using scipy probability distributions.

<img src="media/demo.png" />

## Installation

```bash
uv add marimo-scipy-utils
```

## Functions

[See example notebook here](https://github.com/hbmartin/ai-roi-mcm-npv-marimo/blob/main/ai_roi_mcm_npv.py)

### `abbrev_format(x: float, pos: int | None) -> str`

Format numbers with k/M suffixes for thousands/millions.

**Parameters:**
- `x`: The number to format
- `pos`: The tick position (unused but required by matplotlib formatter interface)

**Returns:**
- `str`: The formatted number string with k/M suffix if applicable

**Example:**
```python
abbrev_format(1500, None)  # Returns "2k"
abbrev_format(2500000, None)  # Returns "3M"
abbrev_format(42.7, None)  # Returns "42.7"
```

### `display_sliders(name: str, sliders: mo.ui.dictionary | mo.ui.slider, invars: dict[str, dict], dist: str | Callable | None = None, descriptions: dict = {}) -> mo.Html`

Display parameter sliders with optional distribution plot.

**Parameters:**
- `name`: Name of the parameter group to display
- `sliders`: Either a single slider or dictionary of sliders for distribution params
- `invars`: Dictionary to store input variable configurations
- `dist`: Distribution to use (string name or callable), required for multi-sliders
- `descriptions`: Optional dict mapping parameter names to descriptions

**Returns:**
- Marimo component displaying the sliders and optional distribution plot

For a single slider, displays it as a constant parameter. For multiple sliders, displays them with a plot of the resulting distribution. Distribution must be specified for multiple sliders, either as a string name matching a scipy distribution or as a callable distribution object.

**Raises:**
- `DistributionConfigurationError`: If dist is None for multiple sliders

### `generate_ranges(distribution: str, ranged_distkwargs: dict) -> dict`

Generate parameter ranges for a probability distribution.

Takes a distribution name and dictionary of parameter ranges, validates the ranges against allowed bounds for that distribution, and returns a merged dictionary with complete range specifications.

**Parameters:**
- `distribution`: Name of the probability distribution (e.g. "normal", "beta")
- `ranged_distkwargs`: Dictionary mapping parameter names to their range specifications. Each range spec should have "lower" and "upper" bounds.

**Returns:**
- `dict`: Complete parameter range specifications with distribution defaults merged with provided ranges

**Raises:**
- `MissingParameterError`: If a required parameter range is not provided
- `ParameterBoundError`: If a provided range exceeds the allowed bounds for a param

### `params_sliders(ranged_distkwargs: dict) -> mo.ui.dictionary`

Create a dictionary of sliders for parameter ranges.

Takes a dictionary of param ranges and creates interactive sliders for each param. The sliders will be bounded by the lower/upper values specified in the ranges dict. The step size and initial value can optionally be specified per param.

**Parameters:**
- `ranged_distkwargs`: Dictionary mapping parameter names to their range specifications. Each range spec should have "lower" and "upper" bounds, and optionally "step" and "value" keys.

**Returns:**
- `mo.ui.dictionary`: A dictionary of marimo slider UI elements, one per parameter. Each slider will be configured according to the parameter's range spec.

**Example:**
```python
ranges = {
    "mean": {"lower": 0, "upper": 100, "step": 1, "value": 50},
    "std": {"lower": 0, "upper": 10}
}
sliders = params_sliders(ranges)
```

## Supported Distributions

Any scipy distribution can be used.

Included helpers following scipy distributions:
- `uniform`: Uniform distribution
- `triang`: Triangular distribution
- `beta`: Beta distribution
- `norm`: Normal distribution

## Dependencies

- marimo
- matplotlib
- scipy
