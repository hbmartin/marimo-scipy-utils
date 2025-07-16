"""Utility functions for creating interactive marimo components.

This module provides functions for creating and configuring interactive UI elements
in marimo notebooks, with a focus on parameter input and visualization:

- abbrev_format: Format numbers with k/M suffixes for thousands/millions
- display_sliders: Create interactive sliders with distribution plots
- generate_ranges: Generate and validate parameter ranges for distributions
- params_sliders: Create a dictionary of parameter sliders
"""

from .marimo_components import (
    abbrev_format,
    display_sliders,
    generate_ranges,
    params_sliders,
)

__all__ = [
    "abbrev_format",
    "display_sliders",
    "generate_ranges",
    "params_sliders",
]
