import json
import pickle
from copy import deepcopy

import numpy as np
from flask import flash, redirect, render_template, request, session, url_for

from simplex_method.core import dualize
from simplex_method.core.dualize import canonical_form
from simplex_method.core.solve.extreme_points import find_extreme_points
from simplex_method.core.solve.optimal_solution import (
    find_optimal_solution,
    target_function,
)
from simplex_method.core.solve.simplex import simplex_solve, tableauos_list
from simplex_method.core.task.task import ConstraintsEnum, Task, TaskTypeEnum

from . import app
from .forms import NumVarConstForm

np.set_printoptions(suppress=True)


@app.route("/", methods=["GET", "POST"])
def index():

    form = NumVarConstForm()
    if request.method == "POST":
        if form.validate_on_submit():
            session["number_of_variables"] = form.number_of_variables.data
            session["number_of_constraints"] = form.number_of_constraints.data
            return redirect(url_for("input_eqs"))
    return render_template("index.html", form=form)


@app.route("/input_eqs", methods=["GET", "POST"])
def input_eqs():
    number_of_variables = session.get("number_of_variables")
    number_of_constraints = session.get("number_of_constraints")
    number_of_var_signs = session.get("number_of_var_signs")
    task = session.get("task")
    constr = [">=" for _ in range(number_of_constraints)]
    sign = []
    if task is not None:
        task = pickle.loads(task)
        with open("values.pickle", "rb") as file:
            task1 = pickle.load(file)
            session["target"] = task1.target_coefs.tolist()
            session["value_right"] = task1.right_part.tolist()
            session["value_left"] = task1.constraints_array.tolist()
            session["constr"] = [x.value for x in task1.constraints]
            session["sign"] = task1.vars_ge_zero.tolist()
            constr = session["constr"]
            sign = session["sign"]

    if request.method == "POST":
        constraint_array = []
        right_part = []
        constraints = []
        for i in range(number_of_constraints):
            cond_value = request.form.get(f"cond_value_{i}")
            right_part.append(0 if cond_value == "" else float(cond_value))

            constraint = request.form.get(f"constraint_{i}")
            match constraint:
                case ">=":
                    constraints.append(ConstraintsEnum.GE)
                case "<=":
                    constraints.append(ConstraintsEnum.LE)
                case "=":
                    constraints.append(ConstraintsEnum.EQ)

            row = []

            for j in range(number_of_variables):
                coef = request.form.get(f"coef_{i}{j}")
                if coef == "":
                    row.append(0)
                else:
                    row.append(float(coef))
            constraint_array.append(row)

        target_coefs = []
        vars_ge_zero = []
        for j in range(number_of_variables):
            targ_coef = request.form.get(f"targ_coef_{j}")
            target_coefs.append(0 if targ_coef == "" else float(targ_coef))

            is_ge = int(request.form.get(f"var_ge_{j}"))
            if is_ge:
                vars_ge_zero.append(j)

        constraint_array = np.array(constraint_array)
        target_coefs = np.array(target_coefs)
        right_part = np.array(right_part)
        vars_ge_zero = np.array(vars_ge_zero)

        task = Task(
            constraints=constraints,
            target_coefs=target_coefs,
            constraints_array=constraint_array,
            right_part=right_part,
            task_type=TaskTypeEnum.MAX,
            vars_ge_zero=vars_ge_zero,
        )

        session["task"] = pickle.dumps(task)
        with open("values.pickle", "wb") as file:
            pickle.dump(task, file)
        return redirect(url_for("results"))
    return render_template(
        "input_eqs.html",
        number_of_variables=number_of_variables,
        number_of_constraints=number_of_constraints,
        number_of_var_signs=number_of_var_signs,
        target=session.get("target"),
        value_right=session.get("value_right"),
        value_left=session.get("value_left"),
        constr=constr,
        sign=sign
    )


@app.route("/results")
def results():
    task = session.get("task")
    if task is not None:
        task = pickle.loads(task)

        dual_task = dualize.dualize(task)
        canonical_task = canonical_form(task)
        task_dict = task.to_dict()
        dual_task_dict = dual_task.to_dict()
        canonical_task_dict = canonical_task.to_dict()
        dual_canonical_form = canonical_form(dual_task)
        dual_canonical_dict = dual_canonical_form.to_dict()

        try:
            extreme_points = find_extreme_points(canonical_task)
            optimal_point, optimal_solution = find_optimal_solution(
                extreme_points, canonical_task
            )

            simplex_point = simplex_solve(canonical_task)

            simplex_solution = target_function(
                canonical_task.target_coefs, simplex_point
            )
            tableauos_list_orig = deepcopy(tableauos_list)
            dual_task_extreme_points = find_extreme_points(dual_canonical_form)
            dual_optimal_point, dual_optimal_solution = find_optimal_solution(
                dual_task_extreme_points,
                dual_canonical_form,
            )

            simplex_point_dual = simplex_solve(dual_canonical_form)

            simplex_dual_solution = target_function(
                dual_canonical_form.target_coefs,
                simplex_point_dual,
            )

            tableauos_list_dual = deepcopy(tableauos_list)
        except Exception:
            flash("Что-то пошло не так, возможно задача поставлена некоректно")
            return redirect(url_for("index"))

        point_method = {
            "point": np.round(optimal_point, 4) if optimal_point is not None else None,
            "solution": (
                np.round(optimal_solution, 4) if optimal_solution is not None else None
            ),
        }
        simplex_method = {
            "point": np.round(simplex_point, 4) if simplex_point is not None else None,
            "solution": (
                np.round(simplex_solution, 4) if simplex_solution is not None else None
            ),
        }

        point_method_dual = {
            "point": (
                np.round(dual_optimal_point, 4)
                if dual_optimal_point is not None
                else None
            ),
            "solution": (
                np.round(dual_optimal_solution, 4)
                if dual_optimal_solution is not None
                else None
            ),
        }
        simplex_method_dual = {
            "point": (
                np.round(simplex_point_dual) if simplex_point_dual is not None else None
            ),
            "solution": (
                np.round(simplex_dual_solution, 4)
                if simplex_dual_solution is not None
                else None
            ),
        }

    else:
        flash("Нужно для начала ввести задачу", "error")
        return redirect(url_for("index"))

    return render_template(
        "results.html",
        task=task_dict,
        dual_task=dual_task_dict,
        canonical_task=canonical_task_dict,
        point_method=point_method,
        simplex_method=simplex_method,
        tableauos_list_orig=tableauos_list_orig,
        tableauos_list_dual=tableauos_list_dual,
        dual_canonical=dual_canonical_dict,
        point_method_dual=point_method_dual,
        simplex_method_dual=simplex_method_dual,
    )
