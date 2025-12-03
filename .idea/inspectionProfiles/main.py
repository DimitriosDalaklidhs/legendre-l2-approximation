# main.py

from typing import Callable, List
import math

from func_f import f
from linsys import solve_linear_system


def legendre_p(n: int, x: float) -> float:
    """
    Computes the Legendre polynomial P_n(x) using the recurrence:
        P_0(x) = 1
        P_1(x) = x
        (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)
    """
    if n == 0:
        return 1.0
    if n == 1:
        return x

    # Initial values P_0 and P_1 are set
    P_nm1 = 1.0
    P_n0 = x
    # Recurrence relation is applied to compute higher degrees
    for k in range(1, n):
        P_np1 = ((2.0 * k + 1.0) * x * P_n0 - k * P_nm1) / (k + 1.0)
        P_nm1, P_n0 = P_n0, P_np1
    return P_n0


def simpson_integration(g: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """
    Performs composite Simpson integration of g(x) on [a, b].
    The number of subintervals n must be even.
    """
    # n is adjusted to the nearest even number
    if n % 2 != 0:
        n += 1

    h = (b - a) / n
    s = g(a) + g(b)

    # Odd-indexed terms are added
    for k in range(1, n, 2):
        s += 4.0 * g(a + k * h)
    # Even-indexed terms are added
    for k in range(2, n, 2):
        s += 2.0 * g(a + k * h)

    return s * h / 6.0


def build_matrix_A(N: int, a: float, b: float) -> List[List[float]]:
    """
    Constructs the matrix A with entries
        A[i][j] = ∫_a^b p_j(x) p_i(x) dx
    where p_k(x) denotes the Legendre polynomial P_k(x).
    """
    A = [[0.0 for _ in range(N + 1)] for _ in range(N + 1)]

    # Each matrix entry is computed by numerical integration
    for i in range(N + 1):
        for j in range(N + 1):
            def integrand(x, i=i, j=j):
                return legendre_p(i, x) * legendre_p(j, x)

            A[i][j] = simpson_integration(integrand, a, b)

    return A


def build_vector_F(N: int, a: float, b: float) -> List[float]:
    """
    Constructs the vector F with entries
        F[i] = ∫_a^b f(x) p_i(x) dx
    """
    F = [0.0 for _ in range(N + 1)]

    # Each entry is computed via numerical integration
    for i in range(N + 1):
        def integrand(x, i=i):
            return f(x) * legendre_p(i, x)

        F[i] = simpson_integration(integrand, a, b)

    return F


def approximate_f(N: int, a: float, b: float):
    """
    Computes the L2-best approximation Π_N f in V_N = span{p_0,...,p_N}
    and evaluates the L2 approximation error.
    """
    # Matrix A and vector F are formed
    A = build_matrix_A(N, a, b)
    F = build_vector_F(N, a, b)

    # The linear system AC = F is solved
    C = solve_linear_system(A, F)

    # Coefficients of the projection are displayed
    print(f"\nBest L2 approximation Π_{N} f(x) = Σ c_k p_k(x)")
    for k, ck in enumerate(C):
        print(f"c_{k} = {ck:.10f}")

    # A function evaluating P_N(x) is defined
    def P_N(x: float) -> float:
        return sum(C[k] * legendre_p(k, x) for k in range(N + 1))

    # The L2 error is calculated by integrating (f - P_N)^2
    def err_integrand(x: float) -> float:
        diff = f(x) - P_N(x)
        return diff * diff

    error_sq = simpson_integration(err_integrand, a, b)
    error = math.sqrt(error_sq)

    print(f"\nL2 error ||f - Π_{N} f||_2 = {error:.10e}")


def main():
    print("L2 projection of function f(x) onto polynomial space V_N with Legendre polynomials.")
    # The user is asked for parameters a, b, and N
    a = float(input("Enter a (e.g. -1): "))
    b = float(input("Enter b (e.g. 1): "))
    N = int(input("Enter N (e.g. 2, 3, 4): "))

    if N < 0:
        raise ValueError("N must be non-negative")

    approximate_f(N, a, b)


if __name__ == "__main__":
    main()
