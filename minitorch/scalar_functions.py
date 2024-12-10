from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the forward pass of a function to the provided scalar values.

        Args:
        ----
        cls (Type[ScalarFunction]): The class of the function to be applied.
        vals (ScalarLike): A variable number of inputs that are either
            `Scalar` objects or values that can be converted into `Scalar` objects.

        Returns:
        -------
        Scalar: A new `Scalar` object representing the result
        of the function applied to the input values. This scalar includes
        a `ScalarHistory` for tracking operations for backpropagation.

        Raises:
        ------
        AssertionError: If the result of the forward pass is not a `float`.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass of the addition function.

        This function returns the sum of two inputs:
            add(a, b) = a + b

        Args:
        ----
            ctx (Context): The context used to store values for backpropagation.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The sum of `a` and `b`.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass of the addition function.

        The derivative of the addition function with respect to both inputs is 1,
        so the gradient is simply the upstream gradient `d_output` for both inputs.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output with respect to some loss.

        Returns:
        -------
            Tuple[float, ...]: The gradient of the inputs `a` and `b` with respect to the loss,

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass of the logarithm function.

        This function returns the natural logarithm of the input:
            log(a) = ln(a)

        Args:
        ----
            ctx (Context): The context used to store values for backpropagation.
            a (float): The input value, which must be greater than 0.

        Returns:
        -------
            float: The natural logarithm of `a`.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass of the logarithm function.

        The derivative of the natural logarithm function is:
            log'(a) = 1 / a

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output with respect to some loss.

        Returns:
        -------
            float: The gradient of the input with respect to the loss.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class EQ(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass of the equality comparison function.

        This function returns 1.0 if `a` is equal to `b`, otherwise 0.0:
            eq(a, b) = 1.0 if a == b else 0.0

        Args:
        ----
            ctx (Context): The context used to store values for backpropagation.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: 1.0 if `a` equals `b`, otherwise 0.0.

        """
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass of the equality comparison function.

        Since the equality function is not differentiable, the gradients
        are always 0 for both inputs.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output with respect to some loss.

        Returns:
        -------
            Tuple[float, ...]: (0.0, 0.0) as the gradients for `a` and `b`.

        """
        return 0.0, 0.0


class LT(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass of the less-than comparison function.

        This function returns 1.0 if `a` is less than `b`, otherwise 0.0:
            lt(a, b) = 1.0 if a < b else 0.0

        Args:
        ----
            ctx (Context): The context used to store values for backpropagation.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: 1.0 if `a` is less than `b`, otherwise 0.0.

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass of the less-than comparison function.

        Since the less-than function is not differentiable, the gradients
        are always 0 for both inputs.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output with respect to some loss.

        Returns:
        -------
            Tuple[float, ...]: (0.0, 0.0) as the gradients for `a` and `b`.

        """
        return 0.0, 0.0


class Exp(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass of the exponential function.

        This function returns the result of the exponential operation:
            exp(a) = e^a

        Args:
        ----
            ctx (Context): The context to store values for backpropagation.
            a (float): The input value.

        Returns:
        -------
            float: The result of applying the exponential function to `a`.

        """
        result = operators.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass of the exponential function.

        The derivative of the exponential function is:
            exp'(a) = exp(a)

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output with respect to some loss.

        Returns:
        -------
            float: The gradient of the input with respect to the loss.

        """
        out: float = ctx.saved_values[0]
        return out * d_output


class Inv(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass of the inverse function.

        This function returns the inverse of the input:
            inv(a) = 1 / a

        Args:
        ----
            ctx (Context): The context to store values for backpropagation.
            a (float): The input value.

        Returns:
        -------
            float: The inverse of `a`.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass of the inverse function.

        The derivative of the inverse function is:
            inv'(a) = -1 / (a^2)

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output with respect to some loss.

        Returns:
        -------
            float: The gradient of the input with respect to the loss.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Mul(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass of the multiplication function.

        This function returns the product of two inputs:
            mul(a, b) = a * b

        Args:
        ----
            ctx (Context): The context to store values for backpropagation.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The product of `a` and `b`.

        """
        ctx.save_for_backward(a, b)
        c = a * b
        return c

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass of the multiplication function.

        The partial derivatives of the multiplication function are:
            d(mul(a, b))/da = b
            d(mul(a, b))/db = a

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output with respect to some loss.

        Returns:
        -------
            Tuple[float, ...]: The gradients of the input values `a` and `b`
            with respect to the loss.

        """
        (
            a,
            b,
        ) = ctx.saved_values
        return b * d_output, a * d_output


class Neg(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass of the negation function.

        This function returns the negation of the input:
            neg(a) = -a

        Args:
        ----
            ctx (Context): The context used to store values for backpropagation.
            a (float): The input value to negate.

        Returns:
        -------
            float: The negated value of `a`.

        """
        return float(-a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass of the negation function.

        The derivative of the negation function is -1:
            neg'(a) = -1

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output with respect to some loss.

        Returns:
        -------
            float: The gradient of the input with respect to the loss.

        """
        return -d_output


class ReLU(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass of the ReLU function.

        The ReLU function is defined as:
            ReLU(a) = max(0, a)

        Args:
        ----
            ctx (Context): The context to store intermediate values for backpropagation.
            a (float): The input value.

        Returns:
        -------
            float: The result of applying the ReLU function to `a`.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass of the ReLU function.

        The derivative of the ReLU function is:
            ReLU'(a) = 1 if a > 0 else 0

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output with respect to some loss.

        Returns:
        -------
            float: The gradient of the input with respect to the loss.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Sigmoid(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass of the sigmoid function.

        The sigmoid function is defined as:
            sigmoid(a) = 1 / (1 + exp(-a))

        Args:
        ----
            ctx (Context): The context to store intermediate values for backpropagation.
            a (float): The input value.

        Returns:
        -------
            float: The result of applying the sigmoid function to `a`.

        """
        result = operators.sigmoid(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass of the sigmoid function.

        The derivative of the sigmoid function is:
            sigmoid'(a) = sigmoid(a) * (1 - sigmoid(a))

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output with respect to some loss.

        Returns:
        -------
            float: The gradient of the input with respect to the loss.

        """
        sigma: float = ctx.saved_values[0]
        return sigma * (1 - sigma) * d_output
