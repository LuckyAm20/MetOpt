import math
import os

import matplotlib
import numpy as np
from flask import flash, redirect, render_template, request, session, url_for
from matplotlib import pyplot as plt

from minimize_function.core import golden_ratio
from minimize_function.core.fibonacci import fibonacci_algorithm
from minimize_function.core.function import FunctionToOptimize
from minimize_function.core.utils import round_by_tol

from . import app

matplotlib.use("Agg")

np.set_printoptions(suppress=True)

NUMBER_OF_PLOT_POINTS = 1000
TOLS = (1e-1, 1e-2, 1e-3)

def optimize_function(func, interval_left, interval_right):
    x_min = np.linspace(interval_left, interval_right, 1000)
    y_min = func(x_min)
    x_min_optimal = x_min[np.argmin(y_min)]
    y_min_optimal = np.min(y_min)
    return x_min_optimal, y_min_optimal

@app.route("/", methods=("GET", "POST"))
def index():
    equation = session.get("equation", None)
    interval_left = session.get("interval_left", 0)
    interval_right = session.get("interval_right", 1)
    number_of_calculations = session.get("number_of_calculations", 1)
    if request.method == "POST":
        equation = request.form.get("equation")
        interval_left = float(request.form.get("interval_left"))
        interval_right = float(request.form.get("interval_right"))
        # number_of_calculations = int(request.form.get("number_of_calcuations"))
        session.update(
            {
                "equation": equation,
                "interval_left": interval_left,
                "interval_right": interval_right,
                "number_of_calculations": number_of_calculations,
            }
        )

    if request.method == "POST":
        return redirect(url_for("plot"))
    context = {
        "equation": equation,
        "interval_left": interval_left,
        "interval_right": interval_right,
        "number_of_calculations": number_of_calculations,
    }
    return render_template("index.html", context=context)


@app.route("/plot", methods=("GET",))
def plot():
    equation = session.get("equation", None)
    interval_left = float(session.get("interval_left", 0))
    interval_right = float(session.get("interval_right", 1))
    # number_of_calculations = session.get("number_of_calculations", 1)

    try:

        func = FunctionToOptimize(equation)
        x_values = np.linspace(interval_left, interval_right, NUMBER_OF_PLOT_POINTS)
        y_values = func(x_values)
        plt.plot(x_values, y_values)
        plt.xlabel("x")
        plt.ylabel(f"f(x) = ${func}$")

        #x_min_optimal = math.exp(1)
        #y_min_optimal = func(x_min_optimal)
        x_min_optimal, y_min_optimal = optimize_function(func, interval_left, interval_right)
        print(f"x_min_optimal={x_min_optimal}\n")
        plt.scatter(x_min_optimal, y_min_optimal, color='red',
                    label=f"Minimum: ({x_min_optimal:.2f}, {y_min_optimal:.2f})")

        plt.scatter(interval_left, func(interval_left), color='green', label=f"a = {interval_left}")
        plt.scatter(interval_right, func(interval_right), color='orange', label=f"b = {interval_right}")
        plt.legend()
        plt.savefig(os.path.join(app.static_folder, "media", "plot.png"))
        plt.close()
        plot_url = os.path.join(app.static_url_path, "media", "plot.png")
        fibonacci_solutions = []
        golden_ratio_solutions = []
        fibonacci_counter = []
        golden_ratio_counter = []
        fibonacci_mistake = []
        golden_ratio_mistake = []
        fibonacci_mistake_y = []
        golden_ratio_mistake_y = []
        for tol in TOLS:
            func.drop_counter()
            golden_ratio_solutions.append(golden_ratio.golden_ratio_algorithm(
                func, interval_left, interval_right, tol
            ))
            golden_ratio_counter.append(func.counter)
            func.drop_counter()
            fibonacci_solutions.append(fibonacci_algorithm(
                func, interval_left, interval_right, tol
            ))
            fibonacci_counter.append(func.counter)

        for i in range(len(TOLS)):
            print(f"fibonacci_solutions_x[{i}] =  {fibonacci_solutions[i]} \n")
            print(f"golden_ratio_solutions_X[{i}] =  {golden_ratio_solutions[i]} \n")
            fibonacci_mistake.append(abs(x_min_optimal - fibonacci_solutions[i]))
            golden_ratio_mistake.append(abs(x_min_optimal - golden_ratio_solutions[i]))
            golden_ratio_mistake_y.append(abs(y_min_optimal - func(golden_ratio_solutions[i])))
            fibonacci_mistake_y.append(abs(y_min_optimal - func(fibonacci_solutions[i])))
            print(f"fibonacci_solutions_y[{i}] =  {func(fibonacci_solutions[i])} \n")
            print(f"golden_ratio_solutions_y[{i}] =  {func(golden_ratio_solutions[i])} \n")
        print(f"fibonacci_mistake_x =  {fibonacci_mistake} \n")
        print(f"golden_ratio_mistake_x =  {golden_ratio_mistake} \n")
        print(f"fibonacci_mistake_y =  {fibonacci_mistake_y} \n")
        print(f"golden_ratio_mistake_y =  {golden_ratio_mistake_y} \n")

        plt.plot(TOLS, fibonacci_counter, label='Fibonacci Calls', marker='o')
        plt.plot(TOLS, golden_ratio_counter, label='Golden Ratio Calls', marker='o')
        plt.xscale('log')
        plt.xlabel('log(Tolerance )')
        plt.ylabel('Number of Function Calls')
        plt.title('Dependency of Function Calls on Tolerance')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(app.static_folder, "media", "plot_calls.png"))
        plt.close()
        plot_calls_url = os.path.join(app.static_url_path, "media", "plot_calls.png")

        plt.plot(TOLS, fibonacci_mistake, label='log of argument  mistake of Fibonacci', marker='o')
        plt.plot(TOLS, golden_ratio_mistake, label='log of argument  mistake of Fibonacci', marker='o')
        plt.plot(TOLS, TOLS, label='log of Tolerance', marker='o')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('log (Tolerance)')
        plt.ylabel('log (Absolute Error)')
        plt.title('Dependency of argument Error on Tolerance')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(app.static_folder, "media", "plot_mistakes.png"))
        plt.close()
        plot_mistake_url = os.path.join(app.static_url_path, "media", "plot_mistakes.png")

        plt.plot(TOLS, fibonacci_mistake_y, label='log of function value mistake of Fibonacci', marker='o')
        plt.plot(TOLS, golden_ratio_mistake_y, label='log of function value mistake of Golden Ratio', marker='o')
        plt.plot(TOLS, TOLS, label='log of Tolerance', marker='o')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('log (Tolerance)')
        plt.ylabel('log (Absolute Error)')
        plt.title('Dependency of function Error on Tolerance')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(app.static_folder, "media", "plot_mistakes_y.png"))
        plt.close()
        plot_mistake_url_y = os.path.join(app.static_url_path, "media", "plot_mistakes_y.png")
    except Exception as e:
        flash(f"Что-то пошло не так. Проверьте корректность функции: {e}")
        plot_url = None
        plot_calls_url = None
        plot_mistake_url = None
        plot_mistake_url_y = None,
    return render_template("plot.html",
                           plot_url=plot_url,
                           plot_calls_url=plot_calls_url,
                           plot_mistake_url=plot_mistake_url,
                           plot_mistake_url_y=plot_mistake_url_y,
                           )


@app.route("/solve", methods=("GET",))
def solve():
    equation = session.get("equation", None)
    interval_left = float(session.get("interval_left", 0))
    interval_right = float(session.get("interval_right", 1))
    # number_of_calculations = session.get("number_of_calculations", 1)
    try:
        func = FunctionToOptimize(equation=equation)

        golden_ratio_solutions = []
        fibonacci_solutions = []
        for tol in TOLS:
            func.drop_counter()
            solution = golden_ratio.golden_ratio_algorithm(
                func, interval_left, interval_right, tol
            )
            golden_ratio_solutions.append(
                {
                    "tol": tol,
                    "solution": solution,
                    "call_count": func.counter,
                    "f": round_by_tol(func(solution), tol),
                }
            )
            func.drop_counter()
            '''solution = fibonacci_algorithm(
                func, interval_left, interval_right, number_of_calculations, tol
            )'''
            solution = fibonacci_algorithm(
                func, interval_left, interval_right, tol
            )
            fibonacci_solutions.append(
                {
                    "tol": tol,
                    "solution": solution,
                    "call_count": func.counter,
                    "f": round_by_tol(func(solution), tol),
                }
            )

        context = {
            "golden_ratio_solutions": golden_ratio_solutions,
            "fibonacci_solutions": fibonacci_solutions,
        }
        return render_template("solve.html", context=context)
    except Exception as e:
        flash(f"Что-то пошло не так. Проверьте корректность функции: {e}")
        return render_template("solve.html")
