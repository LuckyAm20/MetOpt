from typing import List
import numpy as np
import numpy.typing as npt
from enum import Enum
from typing import Dict


class ConstraintsEnum(Enum):
    """# Ограничения
    - GE: >=
    - LE: <=
    - EQ: ="""

    GE = ">="
    LE = "<="
    EQ = "="


class TaskTypeEnum(Enum):
    """# Тип задачи
    - MAX - максимизация F(...) = ... -> max
    - MIN - минимизация F(...) = ... -> min
    """

    MAX = "max"
    MIN = "min"


class Task:
    """# Класс задачи"""

    def __init__(
        self,
        target_coefs: npt.NDArray,
        constraints_array: npt.NDArray,
        right_part: npt.NDArray,
        constraints: List[ConstraintsEnum],
        task_type: TaskTypeEnum,
        vars_ge_zero: npt.NDArray | List,
    ) -> None:
        self.target_coefs = target_coefs
        self.constraints_array = constraints_array
        self.right_part = right_part
        self.constraints = constraints
        self.task_type = task_type
        self.vars_ge_zero = np.array(vars_ge_zero)

    def __str__(self) -> str:
        return (
            f"t: {self.target_coefs}\n"
            f"A: {self.constraints_array}\n"
            f"b: {self.right_part}\n"
            f"c: {[el.value for el in self.constraints]}\n"
            f"x_i >= 0: i: {self.vars_ge_zero}\n"
            f"{self.task_type}"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> Dict:
        constraints = []
        for constraint in self.constraints:
            constraints.append(constraint.value)
        return {
            "target_coefs": np.round(self.target_coefs, 4),
            "constraints_array": np.round(self.constraints_array, 4).tolist(),
            "right_part": np.round(self.right_part, 4),
            "constraints": constraints,
            "task_type": self.task_type.value,
            "vars_ge_zero": self.vars_ge_zero,
        }
