from typing import Sequence

from .module import Parameter
from .scalar import Scalar


class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters


class SGD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to zero.

        Attributes
        ----------
            parameters: A list of model parameters that may have associated
            gradients and derivatives.

        Returns
        -------
            None

        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def step(self) -> None:
        """Update model parameters based on gradients.

        Attributes
        ----------
            parameters: A list of model parameters that may have associated gradients
            or derivatives. Each parameter is updated based on its gradient or derivative.

        Returns
        -------
            None

        """
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.update(Scalar(p.value.data - self.lr * p.value.derivative))
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)
