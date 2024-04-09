from copy import deepcopy
import numpy as np
import numpy.typing as npt
from typing import Tuple
from simplex_method.core.task import Task

tableauos_list = []
visited_states = set()


def create_tableau(c, A, b):
    # 1. Добавляем вектор свободных членов к матрице коэффициентов A
    A = A.copy()

    A = np.hstack((A, b.reshape(-1, 1)))

    # 2. + строка для функции цели
    c = np.hstack((-c, 0))
    # 3. Создаем таблицу
    tableau = np.vstack((A, c))
    tableauos_list.append(tableau)

    return tableau


def is_optimal_plan(tableau):
    cond = np.all(tableau[-1, :-1] >= 0)
    return cond


def find_basis(tableau):
    variables = tableau[:-1, :-1]
    # Подсчитываем количество единиц в каждом столбце
    sums = np.sum(variables, axis=0)
    indices = np.where(sums == 1)[0]

    return indices


def find_pivot_position(tableau):
    negative_coefs = np.where(tableau[-1, :-1] < 0)[0]
    for column in negative_coefs:
        ratios = tableau[:-1, -1] / tableau[:-1, column]  # b / j
        # ratios[tableau[:-1, column] <= 0] = np.inf
        ratios[ratios <= 0] = np.inf

        if not np.all(np.isinf(ratios)):
            break

    if np.all(np.isinf(ratios)):
        return None
    row = np.argmin(ratios)

    print(f'Table: {tableau}')
    tetta = np.min(ratios)

    for column in negative_coefs:
        ratios = tableau[:-1, -1] / tableau[:-1, column]
        ratios[ratios <= 0] = np.inf

        if tetta in ratios:
            print(f'Pivot_vector: {tableau[:-1, -1]}')
            # print(f'Column that : {tableau[:-1, column]}')

        if not np.all(np.isinf(ratios)):
            break

    print(f'Tetta: {tetta}')
    print(f'Basis: {tableau[row, :-1]}')
    print()


    return row, column


def pivot(tableau: npt.NDArray, pivot_position: Tuple[int, int]):
    new_tableau = tableau.copy()
    # new_tableau[:, -1] += 0.001
    i, j = pivot_position
    new_tableau[i] /= tableau[i, j]
    for row in range(len(tableau)):
        if row != i:
            new_tableau[row] -= tableau[row, j] * new_tableau[i]
    return new_tableau


def is_basic(column):
    # Проверяем, является ли столбец базисным
    return np.sum(column) == 1 and np.count_nonzero(column) == 1


def get_solution(tableau):
    # Получаем решение из таблицы
    basic_variables = []
    for col in range(tableau.shape[1] - 1):
        column = tableau[:-1, col]
        if is_basic(column):
            one_index = np.where(column == 1)[0][0]
            basic_variables.append(tableau[one_index, -1])
        else:
            basic_variables.append(0)
    return np.array(basic_variables)


def simplex_solve(task: Task):
    task = deepcopy(task)
    A = task.constraints_array
    b = task.right_part
    c = task.target_coefs
    tableauos_list.clear()
    visited_states.clear()
    tableau = create_tableau(c, A, b)
    tableauos_list.append(tableau)
    # print("init_tableau")
    # print(tableau)
    iterations = 0
    while not is_optimal_plan(tableau):
        print(f'Iteration {iterations}')
        pivot_position = find_pivot_position(tableau)
        if pivot_position is None:
            return None
        tableau_hash = hash_tableau(tableau)
        if tableau_hash in visited_states:
            break
        visited_states.add(tableau_hash)
        tableau = pivot(tableau, pivot_position)
        tableauos_list.append(tableau)
        iterations += 1
    return get_solution(tableau)


def hash_tableau(tableau):
    return hash(tableau.tobytes())