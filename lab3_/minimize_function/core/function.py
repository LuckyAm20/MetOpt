from numbers import Number
from typing import Callable, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T", Number, npt.NDArray)
exp = np.exp
log = np.log
cos = np.cos
sin = np.sin
e = np.e
tg = np.tan
th = np.tanh
sqrt = np.sqrt


class FunctionToOptimize:
    def __init__(
            self,
            equation: str,
            allowed_funcs=None,
    ) -> None:
        if allowed_funcs is None:
            allowed_funcs = {"exp": exp, "log": log, "cos": cos, "sin": sin, "e": e, "tg": tg, "sqrt": sqrt}
        equation = str(equation)

        self.equation = equation.replace("^", "**").replace(" ", "")
        self.func = lambda x: eval(self.equation, {"x": x}.update(allowed_funcs))
        self._counter = 0

    def __str__(self) -> str:
        return self.equation.replace("**", "^")

    def __call__(self, x: T) -> T:
        self._counter += 1
        return self.func(x)

    def drop_counter(self):
        self._counter = 0

    @property
    def counter(self) -> int:
        return self._counter
