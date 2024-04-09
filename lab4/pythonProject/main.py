import os

import numpy as np

import gradf
from configuration import FUNC
import matplotlib.pyplot as plt


def norm(x):
    return (x[0] ** 2 + x[1] ** 2) ** (1 / 2)


x_min, y_min = -0.5, -1
epsilons = [0.1, 0.01, 0.001]
iterations = []
tolerances = []

iter = []
tolerance = []
for i, eps in enumerate(epsilons, start=1):
    M = 3
    alpha = 0.4444
    print(f'Заданная точность = {eps}')
    print('1st ConstDesc:')
    solver = gradf.ConstDesc(func=FUNC)
    solution = solver.get_solution((0.0, 0.0), eps, alpha)
    print('\tsolution: ' + str(solution))
    print('\titeration: ' + str(solver.get_iter_num()))
    iter.append(solver.get_iter_num())
    print('accuracy: ', norm([x_min - solution[0], y_min - solution[1]]))
    tolerance.append(norm([x_min - solution[0], y_min - solution[1]]))
    solver.draw_contoures('1st', i)
iterations.append(iter)
tolerances.append(tolerance)

iter = []
tolerance = []
for i, eps in enumerate(epsilons, start=1):
    print(f'Заданная точность = {eps}')
    print('2nd ConstDesc:')
    solver = gradf.Pshenichnay(func=FUNC)
    solution = solver.get_solution((0.0, 0.0), eps)
    print('\tsolution: ' + str(solution))
    print('\titeration: ' + str(solver.get_iter_num()))
    iter.append(solver.get_iter_num())
    print('accuracy: ', norm([x_min - solution[0], y_min - solution[1]]))
    tolerance.append(norm([x_min - solution[0], y_min - solution[1]]))
    solver.draw_contoures('2nd', i)
iterations.append(iter)
tolerances.append(tolerance)

iter = []
tolerance = []
for i, eps in enumerate(epsilons, start=1):
    print(f'Заданная точность = {eps}')
    print('DFP:')
    solver = gradf.DFP(func=FUNC)
    solution = solver.get_solution((0.0, 0.0), eps)
    print('\tsolution: ' + str(solution))
    print('\titeration: ' + str(solver.get_iter_num()))
    iter.append(solver.get_iter_num())
    x = np.ndarray((1, len(solver.x)))
    y = np.ndarray((1, len(solver.x)))
    print(solver.x)
    for i in range(len(solver.x)):
        x[0, i] = solver.x[i][0]
        y[0, i] = solver.x[i][1]
    print(x[0])
    print(y[0])
    print('accuracy: ', norm([x_min - solution[0], y_min - solution[1]]))
    tolerance.append(norm([x_min - solution[0], y_min - solution[1]]))
    solver.draw_contoures('DFP', i)
    for j in range(len(x[0]) - 1):
        verh = norm([x[0, j + 1] - x_min, y[0, j + 1] - y_min]) ** 2
        niz = norm([x[0, j] - x_min, y[0, j] - y_min])
        print('Соотношение: ', verh / niz)
iterations.append(iter)
tolerances.append(tolerance)

def f(x, y):
    return x ** 2 + x + 3 * y + 1.5 * y ** 2 + 4


f_str = 'x ** 2 + x + 3 * y + 1.5 * y ** 2 + 4'


solution = np.array([-0.5, -1])

f_vec = np.vectorize(f)

x = np.linspace(-10, 7, 100)
y = np.linspace(-8, 5, 100)
x, y = np.meshgrid(x, y)
z = f(x, y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, cmap='plasma', alpha=0.7)
ax.contour(x, y, z, zdir='z', offset=np.min(z), cmap='plasma', levels=20)
ax.scatter(solution[0], solution[1], f(solution[0], solution[1]), color='blue')
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_zlabel('x₃')
fig.colorbar(surf)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(epsilons, iterations[0], label='1st Order iterations', marker='o')
plt.plot(epsilons, iterations[1], label='2nd Order iterations', marker='o')
plt.plot(epsilons, iterations[2], label='DFP iterations', marker='o')
plt.xscale('log')
plt.xlabel('log(Tolerance )')
plt.ylabel('Number of iterations')
plt.title('Dependency of iterations on Tolerance')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(epsilons, tolerances[0], label='1st Order mistakes', marker='o')
plt.plot(epsilons, tolerances[1], label='2nd Order mistakes', marker='o')
plt.plot(epsilons, tolerances[2], label='DFP mistakes', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('log(Tolerance )')
plt.ylabel('log(Mistake )')
plt.title('Dependency of mistakes on Tolerance')
plt.legend()
plt.grid(True)
plt.show()
