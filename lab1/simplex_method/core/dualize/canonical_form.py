from copy import deepcopy
from simplex_method.core.task.task import ConstraintsEnum, Task, TaskTypeEnum
import numpy as np


def canonical_form(task: Task) -> Task:
    task = deepcopy(task)
    task_type = task.task_type
    vars_ge_zero = task.vars_ge_zero
    target_coefs = task.target_coefs
    right_part = task.right_part
    if task_type is TaskTypeEnum.MIN:
        task_type = TaskTypeEnum.MAX
        target_coefs = -target_coefs
    constraints_array = task.constraints_array
    constraints = task.constraints
    non_ge_vars = [i for i in range(len(target_coefs)) if i not in vars_ge_zero]
    target_indexes = [i for i in range(len(target_coefs))]
    canonical_indexes = target_indexes.copy()
    for index_of_non_ge in non_ge_vars:
        ar_col = constraints_array[:, index_of_non_ge: index_of_non_ge + 1]
        ar_col = np.hstack((ar_col, -ar_col))
        constraints_array = np.hstack((constraints_array, ar_col))
        canonical_indexes.append(index_of_non_ge)
        canonical_indexes.append(index_of_non_ge)
        coef = target_coefs[index_of_non_ge]
        target_coefs = np.hstack((target_coefs, np.array([coef, -coef])))
    target_coefs = np.delete(target_coefs, non_ge_vars)
    canonical_indexes = np.delete(canonical_indexes, non_ge_vars)
    index_corresponding = [target_indexes, canonical_indexes]
    constraints_array = np.delete(constraints_array, non_ge_vars, axis=1)
    for i, constraint in enumerate(constraints):
        if constraint is ConstraintsEnum.EQ:
            continue
        new_var = np.zeros((len(constraints), 1))
        if constraint is ConstraintsEnum.GE:
            constraints_array[i] *= -1
            right_part[i] *= -1

        new_var[i] = 1
        constraints_array = np.hstack((constraints_array, new_var))

    constraints = [ConstraintsEnum.EQ] * len(constraints)
    vars_ge_zero = np.array([*range(0, constraints_array.shape[1])])
    if len(target_coefs) < constraints_array.shape[1]:
        target_coefs = np.hstack(
            (
                target_coefs,
                np.zeros(constraints_array.shape[1] - len(target_coefs)),
            )
        )

    for i, rp in enumerate(right_part):
        if rp < 0:
            constraints_array[i] *= -1
            right_part[i] *= -1

    task.vars_ge_zero = vars_ge_zero
    task.constraints_array = constraints_array
    task.target_coefs = target_coefs
    task.constraints = constraints
    task.task_type = task_type
    task.right_part = right_part
    return task, index_corresponding
