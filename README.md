# L2 Projection with Legendre Polynomials

Computes the best L2 approximation of a function f(x) in the polynomial space V_N = span{P_0, ..., P_N}, where P_k are Legendre polynomials.

---

## Method

The approximation Π_N f is found by solving the linear system **AC = F**, where:

- **A[i][j]** = ∫ P_i(x) P_j(x) dx  (Gram matrix)
- **F[i]**    = ∫ f(x) P_i(x) dx

Integration is done via composite Simpson's rule. The system is solved using Gaussian elimination with partial pivoting.

---

## Default Function
```
f(x) = x · sin(4πx)  on  [a, b] = [-1, 1]
```

Defined in `func_f.py` — swap it out to approximate any other function.

---

## Files

| File | Description |
|---|---|
| `func_f.py` | The function f(x) to approximate |
| `linsys.py` | Gaussian elimination solver |
| `main.py` | Legendre polynomials, Simpson integration, projection logic |

---

## Usage
```bash
python main.py
```

You will be prompted for:
- `a`, `b` — the interval endpoints
- `N` — the degree of the polynomial space

**Example:**
```
Enter a (e.g. -1): -1
Enter b (e.g.  1):  1
Enter N (e.g. 2, 3, 4): 4
```

**Output:**
```
Best L2 approximation Π_4 f(x) = Σ c_k P_k(x)
  c_0 = 0.0000000000
  c_1 = ...
  ...

L2 error  ||f - Π_4 f||_2 = 1.2345678901e-03
```

---

## Requirements

Pure Python — no external dependencies. Compatible with Python 3.7+.
