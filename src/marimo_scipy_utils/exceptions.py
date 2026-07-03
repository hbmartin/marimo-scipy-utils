from collections.abc import Mapping


class ParameterValidationError(Exception):
    """Base class for parameter validation errors."""


class MissingParameterError(ParameterValidationError):
    """Raised when a required parameter is missing."""

    def __init__(self, parameter_name: str, ranges: Mapping | None = None):
        if ranges:
            message = f"Missing required parameter: {parameter_name}, must set any `None`s in {ranges}"
        else:
            message = f"{parameter_name}: parameter is not provided"
        super().__init__(message)


class ParameterBoundError(ParameterValidationError):
    """Raised when a parameter value is outside allowed bounds."""

    def __init__(
        self,
        parameter_name: str,
        given_value: float,
        allowed_value: float,
        bound_type: str,
    ):
        lt_gt = "less than" if bound_type == "lower" else "greater than"
        message = f"{parameter_name}: given {bound_type} bound {given_value} is {lt_gt} allowed: {allowed_value}"
        super().__init__(message)


class ParameterRangeError(ParameterValidationError):
    """Raised when a parameter's lower/upper range is invalid."""

    def __init__(self, parameter_name: str, lower: float, upper: float):
        message = (
            f"{parameter_name}: lower bound {lower} must be less than upper bound "
            f"{upper}"
        )
        super().__init__(message)


class UnknownParameterError(ParameterValidationError):
    """Raised when a range is provided for an unsupported parameter."""

    def __init__(
        self,
        parameter_name: str,
        distribution: str,
        supported: list[str],
    ):
        message = (
            f"Unknown parameter for {distribution!r}: {parameter_name!r}, "
            f"supported parameters are: {supported}"
        )
        super().__init__(message)


class DistributionConfigurationError(ParameterValidationError):
    """Raised when distribution configuration is invalid."""

    def __init__(self):
        super().__init__(
            "dist is required when passing a dictionary of parameter sliders; only a single slider (a constant) may omit it",
        )


class UnknownDistributionError(ParameterValidationError):
    """Raised when a distribution name cannot be resolved."""

    def __init__(self, distribution: str, supported: list[str] | None = None):
        if supported:
            message = f"Unknown distribution: {distribution!r}, supported distributions are: {supported}"
        else:
            message = f"Unknown distribution: {distribution!r}, expected the name of a scipy.stats distribution"
        super().__init__(message)
