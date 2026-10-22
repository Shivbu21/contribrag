"""Basic arithmetic helpers for SampleLib."""


def add(a: float, b: float) -> float:
    """Return the sum of a and b."""
    return a + b


def divide(a: float, b: float) -> float:
    """Divide a by b.

    Division by zero is handled here: a ZeroDivisionError with a clear
    message is raised when the denominator is zero.
    """
    if b == 0:
        raise ZeroDivisionError("denominator must not be zero")
    return a / b


class Calculator:
    """A stateful calculator that keeps a running total."""

    def __init__(self) -> None:
        self.total = 0.0

    def multiply(self, x: float) -> float:
        """Multiply the running total by x and return it."""
        self.total *= x
        return self.total
