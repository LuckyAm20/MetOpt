from typing import Callable

import numpy as np

from minimize_function.core.function import T
from minimize_function.core.utils import round_by_tol

GOLDEN_RATIO = (3 - np.sqrt(5)) / 2


def golden_ratio_algorithm(func: Callable[[T], T],
                           a: T, b: T, tol: float = 1e-5) -> T:
    flag = None
    func_eval = 0
    lambda_ =0
    mu =0
    lambda_func = 0
    mu_func = 0
    while abs(a - b) > tol:
        if flag is None:
            lambda_ = a + GOLDEN_RATIO * (b - a)
            mu = a + (1 - GOLDEN_RATIO) * (b - a)
            lambda_func = func(lambda_)
            mu_func = func(mu)
            func_eval += 1
        elif flag:
            mu = lambda_
            lambda_ = a + GOLDEN_RATIO * (b - a)
            mu_func = lambda_func
            lambda_func = func(lambda_)
        else:
            lambda_ = mu
            mu = a + (1 - GOLDEN_RATIO) * (b - a)
            lambda_func = mu_func
            mu_func = func(mu)

        func_eval += 1

        if lambda_func >= mu_func:
            a = lambda_
            flag = False
        else:
            b = mu
            flag = True
    print(f"golden tol = {tol}, func_eval={func_eval},func_counter = {func.counter}\n ")
    print(f"golden tol = {tol}, interval =  [{a},{b}]\n ")
    return round_by_tol((a + b) / 2, tol=tol)
