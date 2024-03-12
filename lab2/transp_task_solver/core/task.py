import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, Literal, Tuple

import numpy as np
import scipy

MIN_VALUE = np.finfo(np.float64).tiny
MAX_VALUE = np.finfo(np.float64).max


@dataclass
class Cell:
    oper: str
    r: int
    c: int


class TransportTask:
    """# Класс транспортной задачи"""

    to_round = 2  # Округлять до

    def __init__(self, a: Iterable, b: Iterable, c: Iterable[Iterable]) -> None:
        r"""# Конструктор транспортной задачи

        Args:
            a (Iterable): Вектор грузов поставщика
            b (Iterable): Вектор потребителей
            c (Iterable[Iterable]): Матрица цен. Размеры должны быть len(a) x len(b),

        Raises:
            ValueError: Если размеры не заданы правильно
        """
        iter(a)  # Проверка, что можно построить итератор
        iter(b)
        iter(c)
        self.a = np.array(a, dtype=np.float64)
        self.b = np.array(b, dtype=np.float64)
        self.c = np.array(c, dtype=np.float64)

        if len(self.a) != self.c.shape[0] or len(self.b) != self.c.shape[1]:
            raise ValueError

        self._transportations = None  # Xij матрица перевозок
        self._basis_vectors = None  # Базисные векторы

        # self._u = None  # Потенциал
        # self._v = None  # Потенциал
        # self._d = None  # Матрица оценок

        self._storage_transportations = []
        self._storage_cycles = []

        self.optimal_point = self.optimal_points_solution = None

    def adjust(self) -> None:
        """# Приведние задачи к закрытому типу.

        Работает по принципу дополнения строки или столбца, если суммарные значения
        потребителей и складов  (sum(a) и (sum(b)) не совпадают, в противном случае
        ничего не меняется
        """
        sum_a = np.sum(self.a)
        sum_b = np.sum(self.b)
        delta_r = abs(sum_a - sum_b)
        if sum_a == sum_b:  # нет необходимости приводить
            return

        if sum_b > sum_a:
            self.a = np.hstack((self.a, delta_r))
            self.c = np.vstack((self.c, np.zeros(self.c.shape[1])))
        else:
            self.b = np.hstack((self.b, delta_r))
            self.c = np.hstack((self.c, np.zeros((self.c.shape[0], 1))))

    def northwest_corner_method(self):
        """# Метод северозаподного угла"""
        self._basis_vectors = np.zeros_like(self.c).astype(bool)
        self._transportations = np.zeros_like(self.c, dtype=np.float64)
        i = j = 0

        while i != self.c.shape[0] and j != self.c.shape[1]:
            left_corner = min(self.a[i], self.b[j])
            self._transportations[i, j] = left_corner
            self.a[i] -= left_corner
            self.b[j] -= left_corner
            self._basis_vectors[i, j] = True
            if self.a[i] == 0:
                i += 1
            if self.b[j] == 0:
                j += 1

    def brute_force_transportation(self):
        """# Метод перебора угловых точек"""
        min_cost = float("inf")
        min_allocation = None

        for corners in self._generate_corners_combinations():
            allocation = np.zeros_like(self.c, dtype=np.float64)
            supply_remaining = self.a.copy()
            demand_remaining = self.b.copy()

            for idx in corners:
                i, j = divmod(
                    idx, self.c.shape[1]
                )  # Получаем индексы строки и столбца из одного числа
                allocation[i, j] = min(supply_remaining[i], demand_remaining[j])
                supply_remaining[i] -= allocation[i, j]
                demand_remaining[j] -= allocation[i, j]

            cost = np.sum(allocation * self.c)
            if cost < min_cost and cost > 0:
                min_cost = cost
                min_allocation = allocation

        return min_allocation, min_cost

    def _generate_corners_combinations(self):
        """# Генерация всех возможных комбинаций угловых точек"""
        num_rows, num_cols = self.c.shape
        for corners in itertools.product(range(num_rows), repeat=num_cols):
            if all(c < num_cols for c in corners):

                yield corners

    def find_first_basis(self) -> Tuple[bool, int, int]:
        indexes = np.where(self._basis_vectors)
        if len(indexes[1] > 0):
            return True, indexes[0][0], indexes[1][0]

        return False, 0, 0

    def _solve_potentials(self) -> bool:
        """# Решение задачи методом потенциалов

        После решения в классе будут перезаписаны атрибут: self._transportations,
        который затем можно будет получить с помощью свойства или специального метода
        """
        # if self._u is None:
        u = np.full_like(self.a, MIN_VALUE)
        # if self._v is None:
        v = np.full_like(self.b, MIN_VALUE)
        status_first_basis, r, c = self.find_first_basis()
        if not status_first_basis:
            return False

        u[r] = 0
        v[c] = self.c[r, c]

        cnt = 1
        while cnt > 0:
            cnt = 0
            for i in range(self._basis_vectors.shape[0]):
                for j in range(self._basis_vectors.shape[1]):
                    if not self._basis_vectors[i, j]:
                        continue

                    if u[i] != MIN_VALUE and v[j] == MIN_VALUE:
                        cnt += 1
                        v[j] = self.c[i, j] - u[i]
                    elif u[i] == MIN_VALUE and v[j] != MIN_VALUE:
                        cnt += 1
                        u[i] = self.c[i, j] - v[j]

        max_d = 0  # Максимальная по модулю оценка
        f_row = f_col = 0  # Начальный вектор в цикле
        # basisYX = np.zeros_like(self.a)
        # basisXY = np.zeros_like(self.b)
        basisYX = [[] for _ in range(len(self.a))]
        basisXY = [[] for _ in range(len(self.b))]

        d = np.zeros_like(self._basis_vectors, dtype=np.float64)  # Вектор потребностей
        for i in range(self.c.shape[0]):
            for j in range(self.c.shape[1]):
                if not self._basis_vectors[i, j]:
                    d[i, j] = self.c[i, j] - u[i] - v[j]
                    if d[i, j] < 0:
                        if max_d < -d[i, j]:
                            max_d = -d[i, j]
                            f_row = i
                            f_col = j
                else:
                    d[i, j] = self.c[i, j] - u[i] - v[j]
                    # basisXY = np.hstack((basisXY, j))
                    # basisYX = np.hstack((basisYX, i))
                    basisYX[i].append(j)
                    basisXY[j].append(i)

        if max_d == 0:
            return False

        # Этап построения цикла
        oper = "+"
        # cycle = []

        point = Cell(oper=oper, r=f_row, c=f_col)
        cycle = [point]
        self._storage_cycles.append(deepcopy(cycle))

        # Индексы по массивам
        idx = np.zeros(sum(self.c.shape)).astype(np.int64)
        idx[1] = -1

        finish = False

        while not finish:
            found = False
            p = len(cycle)

            if not len(cycle):
                for i in range(self.c.shape[0]):
                    for j in range(self.c.shape[1]):
                        if (
                            self._basis_vectors[i, j]
                            and self.transportaitions[i, j] == 0
                        ):
                            self._basis_vectors[i, j] = False
                break

            if cycle[-1].oper == "+":
                # Движение по горизонтали
                row = cycle[-1].r

                for i in range(idx[p] + 1, len(basisYX[row])):
                    idx[p] = i
                    if basisYX[row][i] == cycle[-1].c:
                        continue

                    # Проверяем, что ячейка уже есть в пути
                    used = False
                    for point in cycle:
                        if point.r == row and point.c == basisYX[row][i]:
                            used = True
                            break
                    if used:
                        continue

                    point = Cell(oper="-", r=row, c=basisYX[row][i])
                    cycle.append(point)
                    self._storage_cycles.append(deepcopy(cycle))

                    if (point.r == cycle[0].r or point.c == cycle[0].c) and len(
                        cycle
                    ) > 3:
                        finish = True
                    found = True
                    idx[p + 1] = -1
                    break
            else:
                # Движение по вертикали
                col = cycle[-1].c
                for i in range(idx[p] + 1, len(basisXY[col])):
                    idx[p] = i
                    if basisXY[col][i] == cycle[-1].r:
                        continue

                    # Проверяем, что ячейка уже есть в пути
                    used = False
                    for point in cycle:
                        if point.c == col and point.r == basisXY[col][i]:
                            used = True
                            break
                    if used:
                        continue

                    point = Cell(oper="+", r=basisXY[col][i], c=col)
                    cycle.append(point)

                    self._storage_cycles.append(deepcopy(cycle))

                    if (point.r == cycle[0].r or point.c == cycle[0].c) and len(
                        cycle
                    ) > 3:
                        finish = True

                    found = True
                    idx[p + 1] = -1
                    break
            if not found:
                # Возвращаемся (выкидываем последний элемент)
                cycle = cycle[:-1]
        # Определяем значение пересчета по циклу
        dx = MAX_VALUE
        for point in cycle:
            if point.oper != "-":
                continue
            if self.transportaitions[point.r, point.c] < dx:
                dx = self.transportaitions[point.r, point.c]

        # Пересчет по циклу
        # zero = False
        for point in cycle:

            if point.oper == "+":
                self.transportaitions[point.r, point.c] += dx
            else:
                self.transportaitions[point.r, point.c] -= dx
            self._basis_vectors[point.r, point.c] = (
                self.transportaitions[point.r, point.c] > 0
            )

        # Проверка текущего решения на вырожденность
        cnt = np.sum(self._basis_vectors)

        if cnt < (np.sum(self.c.shape) - 1):
            # Добавляем в базис нулевые клетки
            for _ in range(np.sum(self.c.shape) - 1 - cnt):
                while True:
                    r = np.random.randint(0, len(self.a))
                    c = np.random.randint(0, len(self.b))

                    if not self._basis_vectors[r, c]:
                        self._basis_vectors[r, c] = True
                        break
        return True

    def _solve_points(self):
        """# Решение с помощью метода крайних точек"""
        rows, cols = self.c.shape

        right_part = np.concatenate((self.a, self.b))

        A = np.zeros((rows + cols, rows * cols))

        for i in range(rows):
            A[i, i * cols : (i + 1) * cols] = 1
        for i in range(cols):
            A[i + rows, i::cols] = 1
        # A *= np.random.normal(size=A.shape)
        random_index_to_delete = np.random.randint(0, len(right_part))
        right_part = np.delete(right_part, random_index_to_delete)
        A = np.delete(A, random_index_to_delete, axis=0)
        m, n = A.shape
        if m == n:
            solutions = [np.linalg.solve(A, right_part)]
        else:
            all_column_combinations = itertools.combinations(range(n), m)
            solutions = []
            for comb in all_column_combinations:
                submatrix = A[:, list(comb)]
                if np.linalg.det(submatrix) != 0:
                    x_sub = np.linalg.solve(submatrix, right_part)
                    if np.any(x_sub < 0) or np.any(x_sub >= 1e10):
                        continue
                    x_full = np.zeros(n)
                    x_full[list(comb)] = x_sub
                    solutions.append(x_full)

        if len(solutions) == 0:
            return

        def target_function(solution):
            solution = np.array(solution).reshape(self.c.shape)
            return np.sum(self.c * solution)

        optimal_point = solutions[0]
        optimal_solution = target_function(solution=optimal_point)

        for point in solutions[1:]:
            solution = target_function(solution=point)
            if solution < optimal_solution:
                optimal_solution = solution
                optimal_point = point
        self.optimal_point, self.optimal_points_solution = (
            optimal_point,
            optimal_solution,
        )

    def solve(self, method: Literal["potentials", "points"] = "potentials") -> None:
        """# Интерфейс для получения решений

        В зависимости от аргумента вызывается либо решение методом потенциалов,
        либо решение методом перебора крайних точек.

        Args:
            method (Literal["potentials", "points"]): Если задан литерал `potentials`, то
            будет вызван метод решение потенциалами. Если задан литерал `points`, то будет
            вызван метод решение перебором крайних точек
        """
        if method == "potentials":
            self._storage_transportations.clear()
            self.northwest_corner_method()
            self._storage_transportations.append(self._transportations)
            while self._solve_potentials():
                self._storage_transportations.append(self._transportations)

        elif method == "points":
            self._solve_points()

    @property
    def target_funtion(self) -> float | None:
        """# Целевая функция

        Returns:
            float | None: результат целевой фунции или None, если задача
            инициализирована, но не приведена)"""

        if self._transportations is None:
            return None
        return np.sum(self.c * self._transportations)

    def get_target_function(self) -> float:
        """# Интерфейс получения значения целевой функции для шаблонизатора jinja"""
        return self.target_funtion

    @property
    def transportaitions(self):
        """# Планы перевозок"""
        return self._transportations

    def get_transportations(self):
        """# Интерфейс получения плана перевозок для шаблонизатора jinja"""
        return self.transportaitions

    @property
    def basis(self):
        """Базисные векторы"""
        return self._basis_vectors

    def __str__(self) -> str:
        s = (
            f"F = {self.target_funtion:.2f}"
            if self._transportations is not None
            else "F = None"
        )
        c = (
            e.strip().replace("[", "").replace("]", "")
            for e in f"{np.round(self.c, self.to_round)}".split("\n")
        )
        c = "C = \n" + "\n".join(c)
        a = "a = " + str(np.round(self.a, self.to_round)).replace("[", "").replace(
            "]", ""
        )
        b = "b = " + str(np.round(self.b, self.to_round)).replace("[", "").replace(
            "]", ""
        )
        return f"{s}\n--\n{c}\n--\n{a}\n--\n{b}"

    @property
    def history(self):
        """# История решений в отформатированном виде"""
        result = [
            np.round(el, self.to_round).tolist() for el in self._storage_transportations
        ]
        string_view = []
        for element in result:
            s = []
            for row in element:
                s.append(str(row).replace("[", "").replace("]", ""))
            s = "\n".join(s)
            string_view.append(s)
        return string_view

    def get_history(self):
        """# Интерфейс получения истории решений для шаблонизатора jinja"""
        return self.history

    def get_cycles(self):
        """# Интерфейс для получения истории всех циклов для шаблонизатора jinja и роутеров"""
        return self._storage_cycles
