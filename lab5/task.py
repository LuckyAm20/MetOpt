from math import exp, sqrt


class Task:
    def __init__(self):
        self.dimention = 3
        self.answer = [0, 0, 0]

        # limits with '<= 0' sign
        self.limits = [
            lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 2,
            lambda x: x[0] ** 2 + x[1] ** 2 - 3,
            lambda x: x[1] ** 2 - 3,
        ]

        # gradients for limits with '<= 0' sign
        self.d_limits = [
            lambda x: [2 * x[0],
                       2 * x[1],
                       2 * x[2]],
            lambda x: [2 * x[0],
                       2 * x[1],
                       0],
            lambda x: [0,
                       2 * x[1],
                       0]
        ]

        # limits with '= b' sign
        self.A = [[1, 1, -1]]
        self.b = [-1 / (sqrt(10)) - sqrt(2 / 5)]
        return

    # INITS BY USER
    def f(self, x: list) -> float:
        # RETURN VALUE OF FUNC
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return x1 + x2 + x3 ** 2 + 2 * sqrt(1 + 2 * x1 ** 2 + x2 ** 2)

    # INITS BY USER
    def grad_f(self, x: list) -> list:
        # RETURN VEC OF DIFFS.
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        df_1 = 1 + 4 * x1 / sqrt(2 * x1 ** 2 + x2 ** 2 + 1)
        df_2 = 1 + 2 * x2 / sqrt(2 * x1 ** 2 + x2 ** 2 + 1)
        df_3 = 2 * x3
        return [df_1, df_2, df_3]
