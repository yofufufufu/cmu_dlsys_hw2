from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # keepdims为true，保证Z - Z_max时能正确broadcast
        Z_max = array_api.max(Z, axis=self.axes, keepdims=True)
        # keepdims为true，保证sum_res与Z_max形状一致，避免广播产生非预期的形状
        sum_res = array_api.sum(array_api.exp(Z - Z_max), axis=self.axes, keepdims=True)
        res = array_api.log(sum_res) + Z_max
        # 去掉值为1的维度
        return array_api.squeeze(res)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_Z = node.inputs[0]
        Z_shape = input_Z.shape
        output_shape = out_grad.shape
        # 能够正确broadcast
        if self.axes is None:
            new_grad = out_grad
            new_node = node
        else:
            # 补全形状保证能够broadcast
            new_shape = [1] * len(Z_shape)
            j = 0
            for i in range(len(new_shape)):
                flag = True
                for dele_axes in self.axes:
                    # 说明这一维是被压缩的
                    if i == dele_axes:
                        flag = False
                        break
                # 说明这一维没被压缩
                if flag:
                    new_shape[i] = output_shape[j]
                    j += 1
            new_grad = out_grad.reshape(new_shape)
            new_node = node.reshape(new_shape)
        # 这个函数的导数我推过了，结果就是如下
        return (new_grad * exp(input_Z - new_node)).broadcast_to(Z_shape)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

