import numpy as np
from simplex_method.core.task import Task
from itertools import combinations


def find_extreme_points(task: Task):
    """"""
    constraints_array = task.constraints_array
    right_part = task.right_part

    m, n = constraints_array.shape
    if m == n:
        return [np.linalg.solve(constraints_array, right_part)]

    all_column_combinations = combinations(range(n), m)
    solutions = []
    for comb in all_column_combinations:
        submatrix = constraints_array[:, list(comb)]
        if np.linalg.det(submatrix) != 0:

            x_sub = np.linalg.solve(submatrix, right_part)
            if np.any(x_sub < 0) or np.any(x_sub >= 1e10):
                continue
            x_full = np.zeros(n)
            x_full[list(comb)] = x_sub
            solutions.append(x_full)
    # for j in range(n - size + 1):
    #     submatrix = constraints_array[:, j : j + size]
    #

    return solutions
