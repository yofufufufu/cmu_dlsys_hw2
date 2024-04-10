"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, dtype=dtype))
        if bias:
            # fan_in的值影响到init的结果，所以必须按照作业的提示，取fan_in=out_features，再对结果reshape
            # 对parameter reshape会导致sgd过不了test，但是先对tensor reshape就可以，为什么？
            # self.bias = Parameter(init.kaiming_uniform(out_features, 1, dtype=dtype)).reshape((1, out_features))
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, dtype=dtype).reshape((1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mat_mul_res = X @ self.weight
        if self.bias:
            return mat_mul_res + self.bias.broadcast_to(mat_mul_res.shape)
        return mat_mul_res
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_num = X.shape[0]
        mul_dim = 1
        for dim in X.shape[1:]:
            mul_dim *= dim
        return X.reshape((batch_num, mul_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # 数值稳定的实现，避免logits值太大导致exp计算溢出
        class_num = logits.shape[1]
        batch_num = logits.shape[0]
        mask = init.one_hot(class_num, y)
        batch_res = ops.logsumexp(logits, axes=(1, )) - (logits * mask).sum(axes=(1, ))
        # batch_res.sum() / batch_num, (float32 / int) 会变成float64, 问题出在DivScalar
        # https://numpy.org/doc/stable/reference/generated/numpy.result_type.html
        # https://github.com/dlsyscourse/hw2/issues/11
        return batch_res.sum() / batch_num
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, requires_grad=True))
        self.running_mean = init.zeros(dim, requires_grad=False)
        self.running_var = init.ones(dim, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Train mode
        if self.training:
            batch_num = x.shape[0]
            mean = (x.sum(axes=(0,)) / batch_num)
            x_minus_mean = x - mean.broadcast_to(x.shape)
            var = ((x_minus_mean ** 2).sum(axes=(0,)) / batch_num)
            # 必须使用detach后的mean和var(即使用.data)，否则self.running_mean和self.running_var的require_grad属性会变为True
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
            normed = (x - mean.broadcast_to(x.shape)) / (var.broadcast_to(x.shape) + self.eps) ** 0.5
            return self.weight.broadcast_to(x.shape) * normed + self.bias.broadcast_to(x.shape)
        # Eval mode
        else:
            normed = (x - self.running_mean.broadcast_to(x.shape)) / (self.running_var.broadcast_to(x.shape) + self.eps) ** 0.5
            return self.weight.broadcast_to(x.shape) * normed + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_num = x.shape[0]
        mean = (x.sum(axes=(1, )) / self.dim).reshape((batch_num, 1))
        x_minus_mean = x - mean.broadcast_to(x.shape)
        var = ((x_minus_mean ** 2).sum(axes=(1, )) / self.dim).reshape((batch_num, 1))
        return self.weight.broadcast_to(x.shape) * (x - mean.broadcast_to(x.shape)) / (var.broadcast_to(x.shape) + self.eps) ** 0.5 + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Train mode
        if self.training:
            # 用 * 解构
            # 详见randb的实现，返回的就是一个mask tensor
            mask = init.randb(*x.shape, p=1 - self.p)
            return x / (1 - self.p) * mask
        # Eval mode
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
