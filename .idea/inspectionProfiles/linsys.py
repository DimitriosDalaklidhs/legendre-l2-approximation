# linsys.py

from typing import List

def solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """
    Solves the linear system A x = b using Gaussian elimination
    with partial pivoting. All arguments are plain Python lists.
    """
    # A deep copy is created so the original matrices remain unchanged
    n = len(A)
    M = [row[:] for row in A]
    x = b[:]

    # Forward elimination with partial pivoting is performed
    for k in range(n):
        # The pivot row is identified
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[max_row][k]) < 1e-14:
            raise ValueError("Matrix is singular or nearly singular")

        # Rows in M and x are exchanged if necessary
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
            x[k], x[max_row] = x[max_row], x[k]

        # Entries below the pivot are eliminated
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            M[i][k] = 0.0
            for j in range(k + 1, n):
                M[i][j] -= factor * M[k][j]
            x[i] -= factor * x[k]

    # Back substitution is carried out
    sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = sum(M[i][j] * sol[j] for j in range(i + 1, n))
        sol[i] = (x[i] - sum_ax) / M[i][i]

    return sol
