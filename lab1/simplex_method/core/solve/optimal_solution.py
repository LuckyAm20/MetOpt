from typing import List
from simplex_method.core.task.task import Task
import numpy.typing as npt
import numpy as np


def target_function(target_coefs: npt.NDArray, x: npt.NDArray) -> float:
    if x is None:
        return None
    return np.dot(target_coefs.T, x)


def find_optimal_solution(critical_points: List[npt.NDArray], task: Task):
    # Находим оптимальное решение из тех точек, что были ранее вычислены
    # Так как рассматривается только задача максимизации в канонической форме, то
    # необходимо найти максимальное значение функци цели (target_function)
    if len(critical_points) == 0:
        return None, None
    optimal_point = critical_points[0]
    optimal_solution = target_function(task.target_coefs, optimal_point)
    for point in critical_points[1:]:
        solution = target_function(task.target_coefs, point)
        if solution > optimal_solution:
            optimal_solution = solution
            optimal_point = point

    return optimal_point, optimal_solution
