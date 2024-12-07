"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable


#
# Implementation of a prelude of elementary functions.
def mul(x: float, y: float) -> float:
    """Multiplies two numbers

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The same value as the input.

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers

    Args:
    ----
        x (float): The first value.
        y (float): The second value.

    Returns:
    -------
        float: The addition of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negates a number

    Args:
    ----
        x (float): The input value to negate.

    Returns:
    -------
        float: The negation of x.

    """
    return -x


def lt(x: float, y: float) -> float:
    """Checks if one number is less than the other

    Args:
    ----
        x (float): The first input value.
        y (float): The second input value.

    Returns:
    -------
        bool: True if x < y and False otherwise.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal

    Args:
    ----
        x (float): The first input value.
        y (float): The second input value.

    Returns:
    -------
        bool: True if x is equal to y, False if not.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers

    Args:
    ----
        x (float): The first input value.
        y (flaot): The second input value.

    Returns:
    -------
        float: 'x' if 'x' is larger than 'y', otherwise returns 'y'.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Checks if two numbers are close in value.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: True if the numbers are close (margin of 1e-2), False otherwise.

    """
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    The sigmoid function is defined as:

        sigmoid(x) = 1 / (1 + exp(-x)) for x >= 0
        sigmoid(x) = exp(x) / (1 + exp(x)) for x < 0

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The output of the sigmoid function.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation function.

    The ReLU (Rectified Linear Unit) function is defined as:

        relu(x) = max(0, x)

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The output of the ReLU function.

    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculates the natural logarithm.

    Args:
    ----
        x (float): The value for which to calculate the natural logarithm. Must be greater than 0.

    Returns:
    -------
        float: The natural logarithm of `x`.

    Raises:
    ------
        ValueError: If `x` is less than or equal to 0.

    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential function.

    Args:
    ----
        x (float): The exponent to which `e` is raised.

    Returns:
    -------
        float: The result of `e` raised to the power of `x`.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal of a number.

    Args:
    ----
        x (float): The number for which the reciprocal is calculated.

    Returns:
    -------
        float: The reciprocal of the input number.

    Raises:
    ------
        ValueError: If x is zero.

    """
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Computes the derivative of the natural logarithm multiplied by a given argument.

    Args:
    ----
        x (float): The input value for the logarithm function. Must be greater than zero.
        d (float): The value to multiply the derivative by.

    Returns:
    -------
        float: The product of the derivative of the logarithm and the second argument.

    Raises:
    ------
        ValueError: If x is less than or equal to zero.

    """
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of the reciprocal function multiplied by a given argument.

    Args:
    ----
        x (float): The input value for which the reciprocal derivative is calculated. Must be non-zero.
        d (float): The value to multiply the derivative by.

    Returns:
    -------
        float: The product of the derivative of the reciprocal function and the second argument.

    Raises:
    ------
        ValueError: If x is zero.

    """
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of the ReLU activation function multiplied by a given argument.

    Args:
    ----
        x (float): The input value for which the ReLU derivative is calculated.
        d (float): The value to multiply the derivative by.

    Returns:
    -------
        float: The product of the derivative of the ReLU function and the second argument.

    """
    return d if x > 0 else 0.0


# ## Task 0.3
# Small practice library of elementary higher-order functions.
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a given function to each element of an iterable.

    Args:
    ----
        fn (Callable[[float], float]): A function that takes a float and returns a float.

    Returns:
    -------
        A function that takes a list, applies `fn` to each element, and returns a new list

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Combines elements from two iterables by applying a given function to corresponding elements.

    Args:
    ----
        fn: combine two values

    Returns:
    -------
        Function that takes 2 equally sized lists `ls1` and `ls2`, produce a new list by applying fn(x, y) on each pair of elemnts.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduces an iterable to a single value using a given binary function.

    Args:
    ----
        start: start value $x_0$
        fn: combine two values

    Returns:
    -------
        Function that takes a list `ls` of elements and computes the reduction

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negates all elements in the input list of floats.

    Args:
    ----
        ls (iterable of float): An iterable of floating-point numbers to be negated.

    Returns:
    -------
        iterable of float: A new iterable with all the elements of the input list negated.

    """
    return map(neg)(ls)


def addLists(l1: Iterable[float], l2: Iterable[float]) -> Iterable[float]:
    """Adds corresponding elements from two lists of floats.

    Args:
    ----
        l1 (iterable of float): The first list of floating-point numbers.
        l2 (iterable of float): The second list of floating-point numbers.

    Returns:
    -------
        list of float: A new list containing the sum of corresponding elements from l1 and l2.

    """
    return zipWith(add)(l1, l2)


def sum(l: Iterable[float]) -> float:
    """Sums all elements in a list of floats.

    Args:
    ----
        l (list of float): A list of floating-point numbers to be summed.

    Returns:
    -------
        float: The sum of all elements in the list.

    """
    return reduce(add, 0.0)(l)


def prod(l: Iterable[float]) -> float:
    """Calculates the product of all elements in a list of floats.

    Args:
    ----
        l (iterable of float): A list of floating-point numbers to be multiplied.

    Returns:
    -------
        float: The product of all elements in the list.

    """
    return reduce(mul, 1.0)(l)
