import math
from itertools import combinations


def tsp_dynamic_programming(distance_matrix):
    n = len(distance_matrix)

    # Инициализация массива для хранения длин кратчайших путей и массива для хранения путей
    dp = [[math.inf] * n for _ in range(1 << n)]
    paths = [[None] * n for _ in range(1 << n)]

    # Инициализация начального состояния
    dp[1][0] = 0

    # Вычисление длин кратчайших путей и сохранение путей
    for subset_size in range(2, n + 1):
        for subset in combinations(range(1, n), subset_size - 1):
            mask = sum(1 << i for i in subset)
            for last in subset:
                for curr in subset:
                    if curr != last:
                        new_mask = mask ^ (1 << curr)
                        new_distance = dp[new_mask][curr] + distance_matrix[curr][last]
                        if new_distance < dp[mask][last]:
                            dp[mask][last] = new_distance
                            paths[mask][last] = curr

    # Нахождение минимального пути и его длины
    min_distance = math.inf
    min_path = None
    for last in range(1, n):
        distance = dp[(1 << n) - 1][last] + distance_matrix[last][0]
        if distance < min_distance:
            min_distance = distance
            min_path = last

    # Восстановление пути
    path = [0]
    mask = (1 << n) - 1
    while min_path is not None:
        last = paths[mask][min_path]
        path.append(last)
        mask ^= (1 << min_path)
        min_path = last

    return min_distance, path[::-1]


# Пример использования
distance_matrix = [
    [0, 29, 20, 21],
    [29, 0, 15, 17],
    [20, 15, 0, 28],
    [21, 17, 28, 0]
]

min_distance, optimal_path = tsp_dynamic_programming(distance_matrix)
print("Минимальное расстояние:", min_distance)
print("Оптимальный путь:", optimal_path)
