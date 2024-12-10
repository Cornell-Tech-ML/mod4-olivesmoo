from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, List


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals1 = list(vals)
    vals2 = list(vals)

    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon

    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the given derivative into the variable's existing derivative.

        Args:
        ----
        x (Any): The derivative value to be accumulated into the current derivative.

        Returns:
        -------
        None

        """
        ...

    @property
    def unique_id(self) -> int:
        """Get the unique identifier for this variable.

        Returns
        -------
        int: A unique integer identifier for the variable.

        """
        ...

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf node in the computation graph.

        Returns
        -------
        bool: True if the variable is a leaf node, False otherwise.

        """
        ...

    def is_constant(self) -> bool:
        """Check if the variable is constant.

        A constant variable has no history and does not require gradients.

        Returns
        -------
        bool: True if the variable is constant, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get the parent variables of this variable.

        Returns
        -------
        Iterable[Variable]: An iterable containing the parent variables
                            in the computation graph.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the chain rule for the variable.

        Args:
        ----
        d_output (Any): The derivative to propagate back through the graph.

        Returns:
        -------
        Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each
                                          tuple contains a parent variable
                                          and its corresponding derivative.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    seen = set()
    order: List[Variable] = []

    def dfs(vertex: Variable) -> None:
        if vertex.unique_id in seen or vertex.is_constant():
            return
        if not vertex.is_leaf():
            for parent in vertex.parents:
                if not parent.is_constant():
                    dfs(parent)
        seen.add(vertex.unique_id)
        order.insert(0, vertex)

    dfs(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
        variable (Variable): The right-most variable in the computation graph from which
                             backpropagation starts.
        deriv (Any): The derivative associated with the variable that needs to be propagated
                     backward through the graph.

    Returns:
    -------
        None: This function does not return a value. Instead, it modifies the derivative values
              of each leaf node by writing the results to their derivative attributes through
              `accumulate_derivative`.

    """
    queue = topological_sort(variable)

    derivatives = {}
    derivatives[variable.unique_id] = deriv

    for node in queue:
        d = derivatives[node.unique_id]
        if node.is_leaf():
            node.accumulate_derivative(d)  # add final derivs
        else:
            for var, value in node.chain_rule(d):
                if var.is_constant():
                    continue
                derivatives.setdefault(var.unique_id, 0.0)
                derivatives[var.unique_id] = derivatives[var.unique_id] + value


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Retrieves the saved tensors from the context.

        Returns
        -------
        Tuple[Any, ...]: A tuple containing the saved tensor values.

        """
        return self.saved_values
