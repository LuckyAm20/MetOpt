import numpy as np
import math as m
import task as t
import utils as u
from scipy.optimize import linprog, OptimizeWarning
import warnings
from itertools import combinations


warnings.simplefilter("error", OptimizeWarning)

class SecZoutendijk:
    def get_basis_matrs(self, A : np.ndarray):
        N = A.shape[0]
        M = A.shape[1]

        basis_matrs = []
        basis_combinations_indexes = []
        all_indexes = [i for i in range(M)]

        for i in combinations(all_indexes, N):
            basis_matr = A[:, i]
            if np.linalg.det(basis_matr) != 0:
                basis_matrs.append(basis_matr)
                basis_combinations_indexes.append(i)

        return basis_matrs, basis_combinations_indexes

    def get_possible_vector(self, A : list, b : list):
        N = len(A[0])
        M = len(A)
        
        # find all possible square-shape matrixes in A (M x N) matrix
        if M >= N:
            return []
        else:
            basis_matrs, basis_combinations_indexes = self.get_basis_matrs(np.array(A))

        # watch all possible combinations
        for i in range(len(basis_matrs)):
            solve = np.linalg.solve(basis_matrs[i], b)
            # if bad solution -> check next
            if (len(solve[abs(solve) > 1e+15]) != 0):
                continue

            # if good -> return it
            vec = [0 for i in range(N)]
            for j in range(len(basis_combinations_indexes[i])):
                vec[basis_combinations_indexes[i][j]] = solve[j]
            return vec
    
    def getOnePossibleSolution(self) -> list:
        vector = self.get_possible_vector(self.task.A, self.task.b)
        if len(vector) == 0:
            return []
        return vector

    def __init__(self, task : t.Task):
        # step 1 : create secondary task for finding first step
        self.task = task
        self.dimention = self.task.dimention + 1
        
        # create secondary task restrictions : '+1' for additional variable 'eta'
        # limits with '<= 0' sign
        self.limits = [
            lambda x : x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1/2 - x[3],
            lambda x : x[0] ** 2 + x[1] ** 2 - 3 - x[3],
            lambda x : x[1] ** 2 - 3 - x[3],
        ]

        # gradients for limits with '<= 0' sign
        self.d_limits = [
            lambda x : [2 * x[0],
                        2 * x[1],
                        2 * x[2],
                        -1],
            lambda x : [2 * x[0],
                        2 * x[1],
                        0,
                        -1],
            lambda x : [0,
                        2 * x[1],
                        0,
                        -1]
        ]

        # create '=' restrictions
        self.b = self.task.b
        self.A = list()
        for row in self.task.A:
            newRow = [row[i] if i < self.dimention - 1 else 0 for i in range(self.dimention)]
            self.A.append(newRow)

        # create secondary task target func and his grad
        self.f = lambda x: x[self.dimention - 1]
        self.grad_f = lambda x: [0 if i < self.dimention - 1 else 1 for i in range(self.dimention)]

        # step 2 : find first step of secondary task
        self.x_0 = self.getOnePossibleSolution()
        # print('X0_aux:' + str(self.x_0))
        eta_0 = max([lim(self.x_0) for lim in self.task.limits])
        # print('Eta0_aux:' + str(eta_0))
        self.x_0.append(eta_0)
        return

    # Function which will find additional direction
    def findDirection(self, x_k : list, delta_k : float) -> list():
        nearActiveIndex = u.getNearActiveIndexes(self.limits, x_k, delta_k)
        # print('Jdelta:' + str(nearActiveIndex))
        A_ub = np.zeros(shape=(1 + len(nearActiveIndex), self.dimention + 1))
        b_ub = np.zeros(shape=(1 + len(nearActiveIndex), 1))
        for i in range(len(nearActiveIndex) + 1):
            if i == 0:
                grad = self.grad_f(x_k)
                # print('gradf:' + str(grad))
            else:
                index = nearActiveIndex[i - 1]
                grad = self.d_limits[index](x_k)
                # print('gradphi:' + str(grad))
            for j in range(len(grad)):
                A_ub[i, j] = grad[j]
            A_ub[i, len(grad)] = -1
        # print(A_ub)
        A_eq = np.zeros(shape=(len(self.A), self.dimention + 1))
        b_eq = np.zeros(len(self.A))
        for i in range(len(self.A)):
            for j in range(len(self.A[i])):
                A_eq[i, j] = self.A[i][j]
        # print(A_eq)

        c = np.zeros(self.dimention + 1)
        c[self.dimention] = 1

        bounds = list()
        for i in range(self.dimention):
            bounds.append([-1, 1])
        bounds.append([None, None])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='simplex')

    def ifSecondaryTaskNeedToStop(self, x_k : list) -> bool:
        return x_k[self.dimention - 1] < 0

    def isNeedToFragmentStep(self, x_k : list, alpha_k : float, s_k : list, eta_k : float):
        x_tmp = u.vecSum(x_k, u.vecMul(alpha_k, s_k))
        firstCondition = self.f(x_tmp) <= self.f(x_k) + 1 / 2 * eta_k * alpha_k
        secondCondition = True

        for lim in self.limits:  
            secondCondition = secondCondition and (lim(x_tmp) <= u.ZERO_EPS)

        return (firstCondition and secondCondition) == False

    # Function which will find minimum solving Zoitendijk method
    def solver(self, alpha : float, lambd : float):
        # find first position
        x_k = self.x_0
        alpha_k = alpha
        delta_k = 0.5

        while not self.ifSecondaryTaskNeedToStop(x_k):
            # print('Xk_aux:' + str(x_k))
            # Step 1: find additional direction
            s_and_eta = self.findDirection(x_k, delta_k)
            s_k = list()
            for i in range(len(s_and_eta.x) - 1):
                s_k.append(s_and_eta.x[i])
            eta_k = s_and_eta.x[len(s_and_eta.x) - 1]
            # print('Sk_aux:' + str(s_k))
            # print('Etak_aux:' + str(eta_k))

            # Step 2: find x_k and delta_k
            if eta_k < -delta_k:
                alpha_k = alpha
                while self.isNeedToFragmentStep(x_k, alpha_k, s_k, eta_k):
                    alpha_k *= lambd
                x_k = u.vecSum(x_k, u.vecMul(alpha_k, s_k))
            else:
                delta_k = lambd * delta_k

        # return result of secondary task
        eta_res = x_k.pop(self.dimention - 1)
        # print('X0:' + str(x_k))
        # print('Eta0:' + str(eta_res))
        return x_k, eta_res