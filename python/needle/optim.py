"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            # 注意weight decay的本质是在loss上考虑权重大小，所以他的求导结果也属于权重gradient的一部分
            # PDF上使用weight decay的最终式子只针对不使用momentum的vanilla SGD
            grad = ndl.Tensor(param.grad, dtype='float32').data + self.weight_decay * param.data
            u_t_plus_1 = self.momentum * self.u.get(param, 0) + (1 - self.momentum) * grad
            param.data = param.data - self.lr * u_t_plus_1
            self.u[param] = u_t_plus_1
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            grad = ndl.Tensor(param.grad, dtype='float32').data + self.weight_decay * param.data
            m_t_plus_1 = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * grad
            v_t_plus_1 = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * grad ** 2
            m_t_plus_1_correction = m_t_plus_1 / (1 - self.beta1 ** self.t)
            v_t_plus_1_correction = v_t_plus_1 / (1 - self.beta2 ** self.t)
            param.data = param.data - self.lr * m_t_plus_1_correction / (v_t_plus_1_correction ** 0.5 + self.eps)
            self.m[param] = m_t_plus_1
            self.v[param] = v_t_plus_1
        ### END YOUR SOLUTION
