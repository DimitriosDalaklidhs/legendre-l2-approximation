# func_f.py
import math

def f(x: float) -> float:
    """
    External function f(x).

     ie
        f(x) = x * sin(4πx) on [a, b] = [-1, 1]
    """
    return x * math.sin(4.0 * math.pi * x)
