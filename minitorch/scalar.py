from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if the current scalar is a constant.

        Returns
        -------
        bool: `True` if the current scalar is a constant,
        `False` otherwise.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Retrieves the parent variables of the current scalar.

        Returns
        -------
        Iterable[Variable]: An iterable collection of the parent
        variables associated with the current scalar.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the chain rule for backpropagation.

        This method calculates the derivatives of the inputs to the last
        function in the scalar's computation history using the chain rule.
        It retrieves the necessary context and derivatives from the history
        and pairs each input variable with its corresponding derivative.

        Args:
        ----
        d_output (Any): The gradient of the output with respect to
        the loss, which will be propagated back through the
        computation graph.

        Returns:
        -------
        Iterable[Tuple[Variable, Any]]: A list of tuples, each containing
        a variable and its corresponding derivative.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        derivatives = h.last_fn._backward(ctx=h.ctx, d_out=d_output)
        return list(zip(h.inputs, derivatives))

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Compares if the current scalar is less than another scalar.

        Args:
        ----
        b (ScalarLike): The scalar to compare against.

        Returns:
        -------
        Scalar: A new `Scalar` object representing the result of the comparison.

        """
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Compares if the current scalar is greater than another scalar.

        Args:
        ----
        b (ScalarLike): The scalar to compare against.

        Returns:
        -------
        Scalar: A new `Scalar` object representing the result of the comparison.

        """
        return LT.apply(b, self)

    def __eq__(self, b: ScalarLike) -> Scalar:
        """Compares if the current scalar is equal to another scalar.

        Args:
        ----
        b (ScalarLike): The scalar to compare against.

        Returns:
        -------
        Scalar: A new `Scalar` object representing the result of the comparison.

        """
        return EQ.apply(b, self)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Computes the difference between the current scalar and another scalar.

        Args:
        ----
        b (ScalarLike): The scalar to subtract from the current scalar.

        Returns:
        -------
        Scalar: A new `Scalar` object representing the result of the subtraction.

        """
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> Scalar:
        """Computes the negation of the current scalar.

        Returns
        -------
        Scalar: A new `Scalar` object representing the negation of the current scalar.

        """
        return Neg.apply(self)

    def __add__(self, b: ScalarLike) -> Scalar:
        """Computes the sum of the current scalar and another scalar.

        Args:
        ----
        b (ScalarLike): The scalar to add to the current scalar.

        Returns:
        -------
        Scalar: A new `Scalar` object representing the result of the addition.

        """
        return Add.apply(self, b)

    def log(self) -> Scalar:
        """Applies the natural logarithm function.

        This method computes the natural logarithm of the current scalar,
        which is defined as:
        Log(x) = ln(x)

        Returns
        -------
        Scalar: A new `Scalar` object representing the result
        of applying the natural logarithm function to the current scalar.

        """
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Applies the exponential function.

        This method computes the exponential of the current scalar,
        which is defined as:
        Exp(x) = e^x

        Returns
        -------
        Scalar: A new `Scalar` object representing the result
        of applying the exponential function to the current scalar.

        """
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Applies the Sigmoid function.

        This method computes the sigmoid for the current scalar,
        which is defined as:
            Sigmoid(x) = 1 / (1 + exp(-x))

        Returns
        -------
            Scalar: A new `Scalar` object representing the result
            of applying the Sigmoid function to the current scalar.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Applies the Rectified Linear Unit (ReLU) function.

        This method computes the ReLU for the current scalar,
        which is defined as:
        ReLU(x) = max(0, x)

        Returns
        -------
        minitorch.scalar.Scalar: A new `Scalar` object representing the result
        of applying the ReLU function to the current scalar.

        """
        return ReLU.apply(self)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Parameters
    ----------
    f : Callable
        A function that takes `n` scalar inputs and returns a single `Scalar` output.
    *scalars : Scalar
        A variable number of scalar inputs to be passed into the function `f`.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.unique_id, x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
