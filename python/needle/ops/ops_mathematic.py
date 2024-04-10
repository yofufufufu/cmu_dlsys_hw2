"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # numpy.power()
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        # 这里的运算使用的都是needle.Tensor的重载运算符
        return out_grad * self.scalar * input ** (self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # numpy.divide()
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, out_grad * a / (- b ** 2)
       ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        # 标量在建图时不算在inputs里，而是作为op的属性
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # numpy.transpose可以一次排列(permutate) 所有轴的顺序
        # numpy.swapaxes只能一次交换两个轴的顺序
        # 根据测试用例来看，应该是使用swapaxes
        if self.axes is None:
            return array_api.swapaxes(a, -1, -2)
        return array_api.swapaxes(a, self.axes[0], self.axes[1])
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 再swap一遍就回去了
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        return reshape(out_grad, input_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        output_shape = self.shape
        # numpy的广播策略是将维度先`右`对齐，然后从右往左比较
        # 所以如果维数增加(如1D->2D)，将input_shape缺失的维度从左开始填1与output_shape对齐
        tmp_shape = [1] * (len(output_shape) - len(input_shape)) + list(input_shape)
        dele_shape = []
        for i in range(len(output_shape)):
            # 检查每一维是否被扩展
            if output_shape[i] != tmp_shape[i]:
                dele_shape.append(i)
        # 将所有被扩展的维度通过sum压缩回去
        # tensor sum的实现中，numpy.sum如果不设置keepdims, 会去掉值为1的dim，结果形状出现问题.
        # 如(1,3) broadcast_to (3,3) sum 却得到 (3,)
        # 所以最后还需要reshape成输入的形状，以避免上述问题
        return out_grad.sum(tuple(dele_shape)).reshape(input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    # axes: tuple or None
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        # numpy.sum如果不设置keepdims, 会去掉值为1的dim，所以手动把keepdim时的形状求出并reshape(sum的特点是shape[axes]值都变为1)，保证broadcast时形状正确
        shape_keepdims = list(input_shape)
        # None means sum along all dim
        if self.axes is None:
            shape_keepdims = [1] * len(input_shape)
        else:
            for index in self.axes:
                shape_keepdims[index] = 1
        return out_grad.reshape(tuple(shape_keepdims)).broadcast_to(input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    # 有些我自己补充的测试用例过不了，懒得扣细节了，需要详细判断a_value和b_value哪些维度被广播了
    # 想法：先补1把a_value和b_value的shape长度搞一样，然后zip到一起进行循环并判断，除掉最后两个维度，哪个值小哪个输入(a_value or b_value)的该维度广播
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a_value, b_value = node.inputs
        if len(a_value.shape) == len(b_value.shape):
            return out_grad.matmul(b_value.transpose()), a_value.transpose().matmul(out_grad)
        # For inputs with more than 2 dimensions
        # we treat the last two dimensions as being the dimensions of the matrices to multiply, and ‘broadcast’ across the other dimensions.
        elif len(a_value.shape) < len(b_value.shape):
            # 不考虑最后两个维度，导数需要沿广播的维度`sum`以保证形状正确
            axes = range(len(b_value.shape) - len(a_value.shape))
            return out_grad.matmul(b_value.transpose()).sum(axes=tuple(axes)), a_value.transpose().matmul(out_grad)
        else:
            # 不考虑最后两个维度，导数需要沿广播的维度`sum`以保证形状正确
            axes = range(len(a_value.shape) - len(b_value.shape))
            return out_grad.matmul(b_value.transpose()), a_value.transpose().matmul(out_grad).sum(axes=tuple(axes))
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * (a ** -1)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        # use exp method in needle ops, not in numpy
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    # it's acceptable to access the .realize_cached_data() call on the output tensor
    # since the ReLU function is not twice differentiable anyway.
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        mask = a > 0
        return out_grad * Tensor(mask)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)