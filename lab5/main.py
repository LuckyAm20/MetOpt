import task as t
import zoutendijk as z


def printResult(x_min: list, f_min: float):
    print()
    print("x_min: " + str(x_min))
    print("f(x_min) = " + str(f_min))
    return


def main():
    currentTask = t.Task()

    zoitendijkSolver = z.Zoitendijk(currentTask)

    solution = zoitendijkSolver.solver(1, 0.5)
    printResult(solution, currentTask.f(solution))


main()
