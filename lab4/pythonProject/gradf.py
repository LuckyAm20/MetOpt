import math
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plot
import sympy as sp
from matplotlib import pyplot as plt
from sympy import Matrix, Symbol
from sympy.abc import x, y

FORMULS_FOR_GRANNY = []


def f(x):
    return x[0] ** 2 + x[0] + 3 * x[1] + 4 + 1.5 * x[1] ** 2


def grad_f(x):
    return np.asarray(
        [
            2 * x[0] + 1,
            3 * x[1] + 2
        ]
    )


def norm(x):
    if x is None:
        return float('inf')
    return np.sqrt(x[0] ** 2 + x[1] ** 2)


def fib(n):
    numbers = [1, 1]
    for i in range(2, n):
        num = numbers[i - 2] + numbers[i - 1]
        numbers.append(num)
    return numbers


def fib_min(nums, x_k, grad_k):
    a = 0
    b = 1
    lam = a + nums[-3] / nums[-1] * (b - a)
    mu = a + nums[-2] / nums[-1] * (b - a)
    f_lam = f(x_k - lam * grad_k)
    f_mu = f(x_k - mu * grad_k)
    for i in range(1, len(nums)):
        if f_lam > f_mu:
            a = lam
            lam = mu
            mu = a + nums[-1 - i - 1] / nums[-1 - i] * (b - a)
            if i == len(nums) - 3:
                break
            f_lam = f_mu
            f_mu = f(x_k - mu * grad_k)
        else:
            b = mu
            mu = lam
            lam = a + nums[-1 - i - 2] / nums[-1 - i] * (b - a)
            if i == len(nums) - 3:
                break
            f_mu = f_lam
            f_lam = f(x_k - lam * grad_k)
    if f_lam >= f_mu:
        return (lam + b) / 2
    else:
        return (a + mu) / 2


class Solver:
    x: list
    iters = 0
    fib_iter_num = 20
    func = None
    gradient = None
    find_grad = None
    find_func = None

    def __init__(self, func: str):
        self.x = []
        self.func = sp.sympify(func)
        symb = [x, y]
        self.grad = Matrix([self.func.diff(var) for var in symb])
        self.find_grad = lambda item1: self.grad.evalf(
            subs={x: item1[0], y: item1[1]}
        )
        self.find_func = lambda item1: self.func.evalf(
            subs={x: item1[0], y: item1[1]}
        )

    def get_x_seq(self):
        return self.x.copy()

    def get_iter_num(self):
        return self.iters

    def draw_contoures(self, name, i):
        fig, axis = plot.subplots()
        x = np.ndarray((1, len(self.x)))
        y = np.ndarray((1, len(self.x)))
        for i in range(len(self.x)):
            x[0, i] = self.x[i][0]
            y[0, i] = self.x[i][1]
        x_mesh_min = np.min(x)
        x_mesh_max = np.max(x)
        x_mesh_delta = (x_mesh_max - x_mesh_min) / 10
        x_mesh_min -= x_mesh_delta
        x_mesh_max += x_mesh_delta
        y_mesh_min = np.min(y)
        y_mesh_max = np.max(y)
        y_mesh_delta = (y_mesh_max - y_mesh_min) / 10
        y_mesh_min -= y_mesh_delta
        y_mesh_max += y_mesh_delta
        mesh_dest = max(x_mesh_max - x_mesh_min, y_mesh_max - y_mesh_min)
        x_mesh_max = x_mesh_min + mesh_dest
        y_mesh_max = y_mesh_min + mesh_dest
        x_mesh, y_mesh = np.mgrid[
            x_mesh_min:x_mesh_max:100j, y_mesh_min:y_mesh_max:100j
        ]
        z = np.ndarray(x_mesh.shape)
        for i in range(x_mesh.shape[0]):
            for j in range(x_mesh.shape[1]):
                z[i, j] = f((x_mesh[i, j], y_mesh[i, j]))
        cs = axis.contour(x_mesh, y_mesh, z, levels=8, cmap='plasma')
        axis.plot(x.tolist()[0], y.tolist()[0], 'gX--')
        axis.clabel(cs, colors='black')
        plot.plot(-0.5, -1, 'r*')
        plot.show()
        plt.savefig(os.path.join("media", f"{name}_{i}.png"))
        return fig, axis


class ConstDesc(Solver):
    def __init__(self, func: str):
        super(ConstDesc, self).__init__(func)

    def get_solution(self, x_0, eps, alpha):
        self.x = [np.asarray(x_0)]
        self.iters = 0
        grad = self.find_grad(x_0)
        while grad.norm() >= eps:
            grad = self.find_grad(self.x[-1])
            self.x.append(list((Matrix(self.x[-1]) - alpha * grad)[:, 0]))
            self.iters += 1
            print('x1 = ', self.x[-1][0], 'x2 = ', self.x[-1][1])
            print('step =', alpha)
        return self.x[-1]


class Pshenichnay(Solver):
    hess = None

    def __init__(self, func: str):
        super(Pshenichnay, self).__init__(func)
        self.hess = lambda item1: self.hessian().evalf(
            subs={x: item1[0], y: item1[1]}
        )

    def hessian(self):
        symbols = [x, y]
        n = len(symbols)
        hess = sp.zeros(n, n)
        for i, fi in enumerate(symbols):
            for j, fj in enumerate(symbols):
                hess[i, j] = sp.diff(sp.diff(self.func, fi), fj)
        return hess

    def get_solution(self, x_0, eps):
        self.x = [np.asarray(x_0)]
        self.iters = 0
        grad = self.find_grad(x_0)
        print(grad)
        lambda_ = 1 / 2

        while grad.norm() >= eps:
            p = self.hess(self.x[-1]).LUsolve(-grad)
            alpha = 1.0
            x_new = Matrix(self.x[-1]) + alpha * p
            fx = self.find_func(self.x[-1])

            while self.find_func(
                list(x_new[:, 0])
            ) - fx > alpha * eps * grad.dot(p):
                alpha *= lambda_
                x_new = Matrix(self.x[-1]) + alpha * p

            self.x.append(list(x_new[:, 0]))
            self.iters += 1

            grad = self.find_grad(self.x[-1])

            print('x1 =', self.x[-1][0], 'x2 =', self.x[-1][1])
            print('step =', alpha)

        return self.x[-1]


class DFP(Solver):
    def __init__(self, func: str):
        super().__init__(func)

    def refresh_matrix(self, A):
        dx = Matrix(self.x[-1]) - Matrix(self.x[-2])
        dw = self.find_grad(self.x[-2]) - self.find_grad(self.x[-1])
        return (
                A
                - (dx * dx.T) / dw.dot(dx)
                - A * (dw * dw.T) * A.T
                / dw.dot(A * dw)
        )

    def get_solution(self, x_0, eps):
        self.x = [np.asarray(x_0)]
        self.iters = 0
        fib_nums = fib(self.fib_iter_num)
        grad = self.find_grad(self.x[-1])
        A = sp.eye(2)
        while grad.norm() >= eps:
            grad = self.find_grad(self.x[-1])
            p = A * grad
            alpha = fib_min(fib_nums, Matrix(self.x[-1]), p)
            self.x.append(list((Matrix(self.x[-1]) - alpha * p)[:, 0]))
            self.iters += 1
            if self.iters % 2 == 0:
                A = self.refresh_matrix(A)
        return self.x[-1]