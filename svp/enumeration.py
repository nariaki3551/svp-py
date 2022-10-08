import time

import numpy as np


def dfs(lattice, R, arg_params=None):
    """Depth first enumeration algorithm

    Args:
        lattice (Lattice) : lattice
        R (float) : upper bound of projected vector.
            sqnorm(pi_k(v)) <= R^2 for any k, 0 <= k < lattice.dim
        arg_params (Namespace): parameters from argument parser
    Returns:
        np.array or None: coefficients found in enumeration algorithm for lattice.basis;
                          return None when there is no solution found
    """
    start_time = time.time()

    # floating point
    np_float = np.float64
    np_int = np.int64

    # variable of lattice
    n = lattice.dim
    basis = lattice.basis
    mu = lattice.mu
    B = lattice.B

    # allocations for enum
    squared_R = np.zeros(n, dtype=np_float)
    sigma = np.zeros((n + 1, n), dtype=np_float)
    rho = np.zeros(n + 1, dtype=np_float)
    c = np.zeros(n, dtype=np_float)
    r = np.zeros(n + 1, dtype=np_int)
    v = np.zeros(n, dtype=np_int)
    w = np.zeros(n, dtype=np_int)
    last_nonzero = 1
    k = 0

    # initizalize
    for i in range(n):
        squared_R[i] = R**2
    for i in range(n + 1):
        r[i] = i
    v[0] = 1
    best_coeffs = None

    # main loop
    num_loops = 0

    while True:
        num_loops += 1
        rho[k] = rho[k + 1] + (v[k] - c[k]) ** 2 * B[k]
        if rho[k] < squared_R[n - 1 - k]:
            if k == 0:
                print(f"\tnorm {np.sqrt(rho[0]):.5f}")
                print(f"\tsolution {v}")
                print(f"\tvector {v.T.dot(basis)}")
                print()
                best_coeffs = v.copy()
                for i in range(n):
                    squared_R[i] = min(squared_R[i], rho[0] - 1.0)
            else:
                k -= 1
                r[k] = max(r[k], r[k + 1])
                for i in range(r[k], k - 1, -1):
                    sigma[i, k] = sigma[i + 1, k] + mu[i, k] * v[i]
                c[k] = -sigma[k + 1, k]
                v[k] = round(c[k])
                w[k] = 1
        else:
            k += 1
            if k == n:
                return best_coeffs
            r[k - 1] = k
            if k >= last_nonzero:
                last_nonzero = k
                v[k] += 1
            else:
                if v[k] > c[k]:
                    v[k] -= w[k]
                else:
                    v[k] += w[k]
                w[k] += 1

    print("no solution found")
    return None
