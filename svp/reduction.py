def size_reduce(lattice, i, eta=0.501, U=None):
    """reduce i-th basis vector

    update lattice.basis[i] to be satisfied
    lattice.mu[i, j] < eta for all j, 0 <= j < i

    Args:
        lattice (Lattice): lattice
        i (int): index of reduced basis vector
        eta (double): eta must be greater than 0.5
        U (np.ndarray): inverse matrix of transformation
            such that UC = B, where B is an input basis and C is output basis
    """
    basis = lattice.basis
    mu = lattice.mu

    for j in range(i):
        if abs(mu[i, j]) > eta:
            q = round(mu[i, j])
            basis[i] -= q * basis[j]
            mu[i] -= q * mu[j]
            if U is not None:
                U.T[j] += q * U.T[i]


def LLL(lattice, delta=0.99, eta=0.501, U=None, arg_params=None):
    """LLL reduction

    update lattice.basis to be satisfied
    delta * pi_(k-1)(b*[k-1]) <= pi_(k-1)(b*[k]) for all index k, 1 <= k < lattice.dim,
    where pi_k is a k-th projected function and b*[k] is the k-th gram-schmidt vector

    Args:
        lattice (Lattice): lattice
        delta (double): delta must be in [0.25, 1.0]
        eta (double): eta must be greater than 0.5
        U (np.ndarray): inverse matrix of transformation
            such that UC = B, where B is an input basis and C is output basis
        arg_params (Namespace): parameters from argument parser
    """
    n = lattice.dim
    basis = lattice.basis
    mu = lattice.mu
    B = lattice.B

    k = 1
    while k < n:
        size_reduce(lattice, k, eta=eta, U=U)

        if B[k] >= (delta - mu[k, k - 1] ** 2) * B[k - 1]:
            k += 1
        else:
            basis[[k - 1, k]] = basis[[k, k - 1]]
            mu, B = lattice.set_gso_data()
            if U is not None:
                U.T[[k - 1, k]] = U.T[[k, k - 1]]
            k = max(k - 1, 1)
