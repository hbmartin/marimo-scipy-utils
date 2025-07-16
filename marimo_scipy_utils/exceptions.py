class ParameterValidationError(Exception):
    """Base class for parameter validation errors."""


class MissingParameterError(ParameterValidationError):
    """Raised when a required parameter is missing."""

    def __init__(self, parameter_name: str, ranges: dict | None = None):
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


class DistributionConfigurationError(ParameterValidationError):
    """Raised when distribution configuration is invalid."""

    def __init__(self):
        super().__init__(
            "dist is required for multiple sliders, use _CONST for constants",
        )
