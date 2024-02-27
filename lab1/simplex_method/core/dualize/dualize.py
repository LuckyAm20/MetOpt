from copy import deepcopy
from simplex_method.core.task.task import ConstraintsEnum, Task, TaskTypeEnum
import numpy as np


def _dualize(task: Task):
    task = deepcopy(task)
    # Если задача максимизации, то все неравенства должны быть <= 0
    # Если задача минимизации, то все неравенства должны быть >= 0

    # 1. Просто забираем из объекта задачи все нужные значения
    target_coefs = task.target_coefs
    constraints_array = task.constraints_array
    right_part = task.right_part
    constraints = task.constraints
    vars_ge_zero = task.vars_ge_zero
    task_type = task.task_type

    # 2. Приводим к виду, когда для F() -> max: <= для всех, либо
    #  для F() -> min: >= для всех
    if task_type is TaskTypeEnum.MAX:
        indicies_wrong_inq = [
            i for i in range(len(constraints)) if constraints[i] is ConstraintsEnum.GE
        ]
    else:
        indicies_wrong_inq = [
            i for i in range(len(constraints)) if constraints[i] is ConstraintsEnum.LE
        ]
    right_part[indicies_wrong_inq] *= -1
    constraints_array[indicies_wrong_inq] *= -1

    # 3. Транспонируем матрицу A

    constraints_array = constraints_array.T

    # 4. Меняем местами коэффициенты целевой функции и свободного вектора
    target_coefs, right_part = right_part, target_coefs

    # 5. Меняем тип задачи
    task_type = (
        TaskTypeEnum.MAX if task_type is not TaskTypeEnum.MAX else TaskTypeEnum.MIN
    )

    # 6. Определяем новые ограничения следующим образом:
    # Все переменные с ограничениями дают >= для задачи максимизации (для задачи минимизации <= ) все остальные члены равенства
    new_constraints = []
    for i in range(len(right_part)):
        if i in vars_ge_zero:
            new_constraints.append(
                ConstraintsEnum.GE
                if task_type is TaskTypeEnum.MIN
                else ConstraintsEnum.LE
            )
        else:
            new_constraints.append(ConstraintsEnum.EQ)

    # 7. Определяем новые ограничения на переменные. Все ограничения в исходной задаче дают >= 0 (т.е. vars_ge_zero)
    new_vars_ge_zero = []
    for i, constraint in enumerate(constraints):
        if constraint is not ConstraintsEnum.EQ:
            new_vars_ge_zero.append(i)
    new_vars_ge_zero = np.array(new_vars_ge_zero)

    dual_task = Task(
        target_coefs=target_coefs,
        constraints_array=constraints_array,
        right_part=right_part,
        constraints=new_constraints,
        task_type=task_type,
        vars_ge_zero=new_vars_ge_zero,
    )

    return dual_task


def dualize(task: Task) -> Task:
    """# Функция построения двойственной задачи
    Args:
        task: (Task): Задача

        Returns:
            (Task): Двойственная задача
    """
    return _dualize(task)
