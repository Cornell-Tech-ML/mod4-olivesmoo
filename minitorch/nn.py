from typing import Optional, Tuple

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    input = input.contiguous()

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw

    reshaped = input.view(batch, channel, new_height, kh, new_width, kw)
    permuted = reshaped.permute(0, 1, 2, 4, 3, 5)

    unfolded = permuted.contiguous().view(
        batch, channel, new_height, new_width, kh * kw
    )

    return unfolded, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Applies a 2D average pooling operation over an input tensor.

    Args:
    ----
        input (Tensor): The input tensor of shape `(batch, channel, height, width)`.
        kernel (Tuple[int, int]): The size of the pooling kernel as `(kernel_height, kernel_width)`.

    Returns:
    -------
        Tensor: The output tensor after average pooling with shape
        `(batch, channel, new_height, new_width)`, where `new_height` and
        `new_width` are the reduced spatial dimensions after pooling.

    """
    batch, channel, height, width = input.shape

    unfolded, new_height, new_width = tile(input, kernel)

    pooled = unfolded.mean(dim=4)
    pooled = pooled.view(batch, channel, new_height, new_width)

    return pooled


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Applies a 2D max pooling operation over an input tensor.

    Args:
    ----
        input (Tensor): The input tensor of shape `(batch, channel, height, width)`.
        kernel (Tuple[int, int]): The size of the pooling kernel as `(kernel_height, kernel_width)`.

    Returns:
    -------
        Tensor: The output tensor after max pooling with shape
        `(batch, channel, new_height, new_width)`, where `new_height` and
        `new_width` are the reduced spatial dimensions after pooling.

    """
    batch, channel, height, width = input.shape

    unfolded, new_height, new_width = tile(input, kernel)

    pooled = max(unfolded, dim=4)
    pooled = pooled.view(batch, channel, new_height, new_width)

    return pooled


def softmax(input: Tensor, dim: int) -> Tensor:
    """Computes the softmax of the input tensor along the specified dimension.

    The softmax function transforms the input tensor values into a probability
    distribution, where the values range between 0 and 1, and the sum of the
    values along the specified dimension equals 1.

    Args:
    ----
        input (Tensor): The input tensor containing values to apply the softmax function to.
        dim (int): The dimension along which to compute the softmax. Must be a valid dimension of the input tensor.

    Returns:
    -------
        Tensor: The output tensor with the same shape as the input tensor, where
        values along the specified dimension are normalized into probabilities.

    """
    exp_tensor = input.exp()
    sum_exp = exp_tensor.sum(dim)

    softmax_output = exp_tensor / sum_exp

    return softmax_output


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Computes the logarithm of the softmax function for the input tensor along the specified dimension.

    Args:
    ----
        input (Tensor): The input tensor for which the log-softmax is computed.
        dim (int): The dimension along which to compute the log-softmax. Must be a valid dimension of the input tensor.

    Returns:
    -------
        Tensor: A tensor of the same shape as the input tensor, containing the log-softmax
        values along the specified dimension.

    """
    max_val = max(input, dim)
    exp_input = (input - max_val).exp()
    sum_exp = exp_input.sum(dim=dim)
    log_sum_exp = sum_exp.log()

    return input - log_sum_exp - max_val


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Applies dropout to the input tensor by randomly setting elements to zero with probability `p`.

    Args:
    ----
        input (Tensor): The input tensor to which dropout will be applied.
        p (float): The probability of setting an element to zero. Should be a value in the range [0, 1].
        ignore (bool, optional): If True, skips applying dropout and returns the input tensor as-is. Defaults to False.

    Returns:
    -------
        Tensor: A tensor of the same shape as the input, where elements are scaled by a binary mask.

    """
    if ignore:
        return input
    mask = rand(input.shape)
    mask = mask > p
    return input * mask


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor."""
    max_tensor = input.f.max_reduce(input, dim)
    # one_hot_tensor = zeros(input.shape)
    # for i in range(input.shape[dim]):
    #     mask = input.f.select(dim, i) == max_tensor
    return input == max_tensor


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Performs the forward pass of the max reduction operation along a specified dimension.

        Args:
        ----
            ctx (Context): A context object used to store information for the backward computation.
            a (Tensor): The input tensor on which the max reduction operation is performed.
            dim (Tensor): A tensor containing the dimension along which the reduction is computed.

        Returns:
        -------
            Tensor: A tensor containing the maximum values along the specified dimension.

        """
        ctx.save_for_backward(a, dim)
        return a.f.max_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradients."""
        (a, dim) = ctx.saved_values
        # print("a", a, "dim", dim)
        return argmax(a, int(dim.item())) * grad_output, 0.0


def max(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Computes the maximum value of a tensor, either across the entire tensor or along a specified dimension.

    Args:
    ----
        input (Tensor): The input tensor on which the maximum operation is performed.
        dim (Optional[int]): The dimension along which to compute the maximum values. If `None`, the
            maximum is computed over the entire tensor.

    Returns:
    -------
        Tensor: A tensor containing the maximum values. If `dim` is specified, the result contains
        the maximum values along the specified dimension. If `dim` is `None`, a single value is returned.

    """
    if dim is None:
        return Max.apply(input.contiguous().view(input.size), input._ensure_tensor(0))
    else:
        return Max.apply(input, input._ensure_tensor(dim))
