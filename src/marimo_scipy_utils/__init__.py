"""Utility functions for creating interactive marimo components.

This module provides functions for creating and configuring interactive UI elements
in marimo notebooks, with a focus on parameter input and visualization:

- abbrev_format: Format numbers with k/M/B suffixes
- display_sliders: Create interactive sliders with distribution plots
- generate_ranges: Generate and validate parameter ranges for distributions
- params_sliders: Create a dictionary of parameter sliders
- resolve_distribution: Resolve a distribution name to a scipy.stats callable
- sample_invars: Sample all configured input variables (e.g. for Monte Carlo)
- InputVar: A configured input variable (constant or distribution)
- RangeSpec: Typed range specification for a distribution parameter
"""

from .exceptions import (
    DistributionConfigurationError,
    MissingParameterError,
    ParameterBoundError,
    ParameterRangeError,
    ParameterValidationError,
    UnknownDistributionError,
    UnknownParameterError,
)
from .marimo_components import (
    SCIPY_DISTRIBUTIONS,
    InputVar,
    RangeSpec,
    abbrev_format,
    display_sliders,
    generate_ranges,
    params_sliders,
    resolve_distribution,
    sample_invars,
)

__all__ = [
    "SCIPY_DISTRIBUTIONS",
    "DistributionConfigurationError",
    "InputVar",
    "MissingParameterError",
    "ParameterBoundError",
    "ParameterRangeError",
    "ParameterValidationError",
    "RangeSpec",
    "UnknownDistributionError",
    "UnknownParameterError",
    "abbrev_format",
    "display_sliders",
    "generate_ranges",
    "params_sliders",
    "resolve_distribution",
    "sample_invars",
]
