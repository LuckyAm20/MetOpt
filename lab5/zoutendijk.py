import numpy as np
import math as m
import task as t
import utils as u
from scipy.optimize import linprog, OptimizeWarning
import warnings
from start_zoutendijk import SecZoutendijk as sz
from math import exp, sqrt

warnings.simplefilter("error", OptimizeWarning)


def norm_sub(vec1, vec2):
    norm = 0
    for i in range(len(vec1)):
        norm += (vec1[i] - vec2[i]) ** 2
    return sqrt(norm)


class Zoitendijk:
    def __init__(self, task: t.Task):
        self.task = task
        return

    # Function which will return first approximation of solution
    def firstApproxim(self) -> list:
        secZ = sz(self.task)
        x, eta = secZ.solver(1, 0.5)
        return x

    # Function which will find additional direction
    def findDirection(self, x_k: list, delta_k: float) -> list():
        nearActiveIndex = u.getNearActiveIndexes(self.task.limits, x_k, delta_k)

        A_ub = np.zeros(shape=(1 + len(nearActiveIndex), self.task.dimention + 1))
        b_ub = np.zeros(shape=(1 + len(nearActiveIndex), 1))
        for i in range(len(nearActiveIndex) + 1):
            if i == 0:
                grad = self.task.grad_f(x_k)
            else:
                index = nearActiveIndex[i - 1]
                grad = self.task.d_limits[index](x_k)
            for j in range(len(grad)):
                A_ub[i, j] = grad[j]
            A_ub[i, len(grad)] = -1

        A_eq = np.zeros(shape=(len(self.task.A), self.task.dimention + 1))
        b_eq = np.zeros(len(self.task.A))
        for i in range(len(self.task.A)):
            for j in range(len(self.task.A[i])):
                A_eq[i, j] = self.task.A[i][j]

        c = np.zeros(self.task.dimention + 1)
        c[self.task.dimention] = 1

        bounds = list()
        for i in range(self.task.dimention):
            bounds.append([-1, 1])
        bounds.append([None, None])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, 'simplex')

    def ifZoitendijkMethodNeedToStop(self, x_min: list, delta_k: float, eta_k: float) -> bool:
        activeLims = u.getActiveIndexes(self.task.limits, x_min)
        allRestrictionDeltas = list()
        c = list()
        for lim in range(len(self.task.limits)):
            if lim in activeLims:
                continue
            else:
                c.append(lim)
                allRestrictionDeltas.append(self.task.limits[lim](x_min))
        delta_0k = -max(allRestrictionDeltas)
        print()
        print("$\ delta_{k} =", end=' \\num{')
        print(str(delta_k), end='}$ , ')
        print("$\ delta_{0k} =", end=' \\num{')
        print(str(delta_0k), end='}$ \\\\')
        print()
        print("Ограничения =", end=' ')
        print(str(allRestrictionDeltas), end=' \\\\')
        return (abs(eta_k) < u.ZERO_EPS and delta_k < delta_0k)

    def isNeedToFragmentStep(self, x_k: list, alpha_k: float, s_k: list, eta_k: float):
        x_tmp = u.vecSum(x_k, u.vecMul(alpha_k, s_k))
        firstCondition = self.task.f(x_tmp) <= self.task.f(x_k) + 1 / 2 * eta_k * alpha_k
        secondCondition = True

        for lim in self.task.limits:
            res = lim(x_tmp)
            secondCondition = secondCondition and (res <= 0)

        return (firstCondition and secondCondition) == False

    # Function which will find minimum solving Zoitendijk method
    def solver(self, alpha: float, lambd: float):
        # find first position
        x_opt = [-1 / (sqrt(10)), - sqrt(2 / 5), 0]
        f_x_opt = sqrt(5 / 2)
        x_k = self.firstApproxim()
        k = 0
        alpha_k = alpha
        delta_k = 0.25

        while True:
            print('\nIteration ' + str(k), end=' \\\\')
            # Step 1: find additional direction
            s_and_eta = self.findDirection(x_k, delta_k)
            s_k = list()
            for i in range(len(s_and_eta.x) - 1):
                s_k.append(s_and_eta.x[i])
            eta_k = s_and_eta.x[len(s_and_eta.x) - 1]

            # Step 2: find x_k and delta_k
            if eta_k < -delta_k:
                alpha_k = alpha
                while self.isNeedToFragmentStep(x_k, alpha_k, s_k, eta_k):
                    alpha_k *= lambd
                x_k = u.vecSum(x_k, u.vecMul(alpha_k, s_k))
            else:
                delta_k = lambd * delta_k

            # if need to leave - leave)
            if self.ifZoitendijkMethodNeedToStop(x_k, delta_k, eta_k):
                break
            k += 1

            # print(str(eta_k), end=' ')
            # print(str(u.ZERO_EPS),  end=' ')
            print()
            print("$x_{k} =", end=' ')
            print(str(x_k), end='$\\\\ ')
            print("$f(x_{k}) =", end=' \\num{')
            print(str(self.task.f(x_k)), end='}$\\\\ ')
            print()
            print("$||x_{k} - x_{opt}|| = ", end=' \\num{')
            print(str(norm_sub(x_k, x_opt)), end=' }$\\\\ ')
            print(" $|f(x_k)-f(x_{opt})| =", end=' \\num{')
            print(str(abs(f_x_opt - self.task.f(x_k))), end='}$\\\\')

        return x_k
