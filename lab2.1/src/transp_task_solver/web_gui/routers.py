import numpy as np
from click import style
from flask import flash, redirect, render_template, request, session, url_for

from transp_task_solver.core.task import TransportTask

from . import app

np.set_printoptions(suppress=True)


@app.route("/", methods=["GET", "POST"])
def index():
    n_a = session.get("n_a")
    n_b = session.get("n_b")
    if n_a is not None and n_b is not None:
        nm = {"n": n_a, "m": n_b}
    else:
        nm = {"n": 1, "m": 1}
    if request.method == "POST":
        n_a = int(request.form.get("numOfA"))
        n_b = int(request.form.get("numOfB"))
        session["n_a"] = n_a
        session["n_b"] = n_b
        session.pop("task", None)

        return redirect(url_for("input_task"))

    return render_template("index.html", nm=nm)


@app.route("/input_task", methods=["GET", "POST"])
def input_task():
    n_a = session.get("n_a")
    n_b = session.get("n_b")
    task = session.get("task")
    nm = {"n": n_a, "m": n_b}

    if request.method == "POST":
        a = []
        b = []
        c = []

        for i in range(n_a):
            a.append(float(request.form.get(f"a{i}")))
            c.append([])
            for j in range(n_b):
                c[i].append(float(request.form.get(f"c{i}{j}")))
                if i == n_a - 1:
                    b.append(float(request.form.get(f"b{j}")))

        task = {"a": a, "b": b, "c": c}

        session["task"] = task
        return redirect(url_for("closed_form_of_task"))

    return render_template("input_task.html", nm=nm, task=task)


@app.route("/closed_form_of_task")
def closed_form_of_task():
    saved_task = session.get("task")
    if saved_task is None:
        return render_template("closed_form_task.html", task=None)
    # n_a = session.get("n_a")
    # n_b = session.get("n_b")
    task = TransportTask(saved_task["a"], saved_task["b"], saved_task["c"])
    task.adjust()  # Приведение к закрытой форме
    n_a, n_b = task.c.shape
    nm = {"n": n_a, "m": n_b}
    return render_template("closed_form_task.html", task=task, nm=nm)


@app.route("/potentials_solve")
def potentials_solve():
    saved_task = session.get("task")
    if saved_task is None:
        return render_template("potentials_solve.html", task=None)
    task = TransportTask(saved_task["a"], saved_task["b"], saved_task["c"])
    task.adjust()  # Приведение к закрытой форме, если вдруг не приведено
    task.solve(method="potentials")
    n_a = len(task.a)
    n_b = len(task.b)
    nm = {"n": n_a, "m": n_b}
    cells_styles = []
    for cycle in task.get_cycles():
        styles = [[None] * n_b for _ in range(n_a)]
        for point in cycle:
            styles[point.r][point.c] = point.operator
        cells_styles.append(styles)
    return render_template(
        "potentials_solve.html",
        task=task,
        nm=nm,
        cells_styles=cells_styles,
    )


@app.route("/points_solve")
def points_solve():
    saved_task = session.get("task")
    if saved_task is None:
        return render_template("points_solve.html", task=None)

    task = TransportTask(saved_task["a"], saved_task["b"], saved_task["c"])
    task.adjust()  # Приведение к закрытой форме, если вдруг не приведено
    task.solve(method="points")
    task.optimal_point = task.optimal_point.reshape(task.c.shape)
    n_a = len(task.a)
    n_b = len(task.b)
    nm = {"n": n_a, "m": n_b}
    return render_template("points_solve.html", task=task, nm=nm)
