import numpy as np


def set_presicion(numbers: np.array):
    string_array = numbers.astype(str)

    # Проходимся циклом по всем элементам и добавляем нули
    for i in range(len(string_array)):
        decimal_index = string_array[i].find('.')
        if decimal_index != -1 and len(string_array[i][decimal_index + 1:]) < 4:
            # Добавляем нули после точки
            string_array[i] += '0' * (4 - len(string_array[i][decimal_index + 1:]))

    return string_array


def create_in_original_basis(optimal_solution, index_corresponding):
    if optimal_solution is None:
        return None
    optimal_basis_solution = [0] * len(index_corresponding[0])
    length = len(index_corresponding[1])
    i = 0
    while i < len(index_corresponding[1]):
        if i + 1 < length:
            if index_corresponding[1][i] == index_corresponding[1][i + 1]:
                optimal_basis_solution[index_corresponding[1][i]] = optimal_solution[i] - optimal_solution[i + 1]
                i += 1
            else:
                optimal_basis_solution[index_corresponding[1][i]] = optimal_solution[i]
        elif i == length - 1:
            optimal_basis_solution[index_corresponding[1][i]] = optimal_solution[i]
        i += 1
    print(optimal_basis_solution)
    return np.array(optimal_basis_solution)


def function_value(task, coefs):
    value = [0] * 1
    for i in range(len(task.target_coefs)):
        value[0] += task.target_coefs[i] * coefs[i].astype('float64')
    return np.array(value)


def convert_to_arr(point):
    val = [0] * 1
    val[0] = point
    return np.array(val)
