import numpy as np


def round_by_tol(value, tol):
    return np.round(value, len(str(int(1 / tol))) - 1)
