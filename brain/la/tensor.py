"""Tensor

    This module contains Tensor classes for handling the tensor concept as a mathematical object that generalizes the
    concepts of scalars, vectors, and matrices to higher dimensions. A tensor can be thought of as a multidimensional
    array of numbers, where each element is indexed by a set of integers.

    Example usage:
        >>> CuPyTensor((1, 2))
"""

# region Import dependencies--------------------------------------------------------------------------------------------
from typing import Any
import cupy as cp
# endregion Import dependencies


# region Base Classes---------------------------------------------------------------------------------------------------
class Tensor(cp.ndarray):
    """Tensor

        This class defines the CuPy-based tensor base object class to handle tensors with any dimension. It is inherited
        over the :class:`cp.ndarray` class from CuPy library.

        Attributes
            requires_grad:
                A :type:`bool` that specifies whether the tensor to be considered in the operation of automatic
                differentiation or not.
            grad:
                A :class:`Tensor` that specifies the gradients for the current tensor. This attribute is `None` by
                default and becomes a :class:`CuPyTensor` the first time a call to :meth:`backward()` computes for the
                current tensor.
            grad_fn:
                It is used to keep track of the function that created the tensor through a chain of operations that
                involved computing gradients. When a tensor is created through an operation that involves computing
                gradients, AIBrain automatically adds a `grad_fn` attribute to the tensor that references the function
                that created the tensor. This function is responsible for computing the gradients of the tensor with
                respect to its inputs.
    """

    # region Constructor and Destructor---------------------------------------------------------------------------------
    def __init__(self, *args, requires_grad: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_grad: bool = requires_grad
        self.grad: Tensor = None
        self.grad_fn = None
    # endregion Constructor and Destructor

    # region Methods----------------------------------------------------------------------------------------------------
    def backward(self):
        """Compute the gradient of current tensor

        :return:
        """
    # endregion Methods
# endregion Base Classes
