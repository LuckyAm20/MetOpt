from typing import Any, Callable, Generator

from minimize_function.core.function import T
from minimize_function.core.utils import round_by_tol


def fibonacci_algorithm(
        func: Callable[[T], T], a: T, b: T, tol: float = 1e-5
) -> T:

    fibo = [1, 1]
    while fibo[-1] < (b - a) / tol:
        fibo.append(fibo[-1] + fibo[-2])

    n = len(fibo) - 1
    lam = a + fibo[n - 2] / fibo[n] * (b - a)
    mu = a + fibo[n - 1] / fibo[n] * (b - a)

    lam_func = func(lam)
    mu_func = func(mu)
    func_evals = 2

    for k in range(1, n - 1):
        if lam_func > mu_func:
            a = lam
            lam = mu
            mu = a + fibo[n - k - 1] / fibo[n - k] * (b - a)
            lam_func = mu_func
            mu_func = func(mu)
        else:
            b = mu
            mu = lam
            lam = a + fibo[n - k - 2] / fibo[n - k] * (b - a)
            mu_func = lam_func
            lam_func = func(lam)

        func_evals += 1

    if lam_func <= mu_func:
        b = mu
    else:
        a = lam
    print(f"fibanacci tol = {tol}, func_eval={func_evals},func_counter = {func.counter}\n ")
    print(f"fibanacci tol = {tol}, interval =  [{a},{b}]\n ")
    return round_by_tol((a + b) / 2, tol=tol)

