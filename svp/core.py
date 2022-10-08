import math
import random

import numpy as np
import scipy


class Lattice:
    """Lattice manager

    Attributes:
        basis (np.ndarray): lattice basis square matrix
        mu (np.ndarray): coefficients matrix of the gram-schmidt orthogonalization of basis
        B (np.ndarray): B[i] is squared norm of the i-th gram-shmidt vector
        dim (int): dimension of this lattice
        GH (np.float): guass heuristics, estimated value of the shortest vector norm in this lattice
    """

    def __init__(self, basis):
        """
        Args:
            basis (np.ndarray): lattice basis square matrix
        """
        assert basis.ndim == 2 and basis.shape[0] == basis.shape[1]
        self.basis = basis
        self.mu = None
        self.B = None
        self.set_gso_data()  # set self.mu and self.B

    @classmethod
    def create_from_file(cls, instance_file):
        """
        Args:
            instance_file (str): svp challenge instance file path

        Returns:
            Lattice object whose basis is read from instance_file
        """
        with open(instance_file, "r") as f:
            basis = []
            for line in f.readlines():
                line = line.strip().replace("[", "", -1).replace("]", "", -1)
                if line:
                    basis.append(list(map(int, line.split())))
        return cls(np.array(basis, dtype=np.float64))

    @property
    def dim(self):
        """
        Returns:
            int: dimension of this lattice
        """
        return self.basis.shape[0]

    @property
    def GH(self):
        """
        Returns:
            np.float: gauss heuristics value
        """
        log_volume = np.sum(np.log(self.B)) * 0.5
        log_unit_sphere_volume = self.dim * np.log(np.pi) * 0.5 - np.log(
            scipy.special.gamma(self.dim * 0.5 + 1.0)
        )
        return np.exp((log_volume - log_unit_sphere_volume) / self.dim)

    def set_gso_data(self):
        """set self.mu and self.B

        Returns:
            (np.ndarray, np.ndarray): (self.mu, self.B)
        """
        basis = self.basis
        mu = np.identity(self.dim, dtype=np.float64)
        B = np.zeros(self.dim, dtype=np.float64)
        r = np.zeros(self.dim, dtype=np.float64)

        for i in range(self.dim):
            for j in range(i):
                r[j] = basis[i].dot(basis[j]) - np.sum(mu[j, :j] * r[:j])
                mu[i, j] = r[j] / B[j]
            B[i] = basis[i].dot(basis[i]) - np.sum(mu[i, :i] * r[:i])

        self.mu = mu
        self.B = B
        return mu, B

    def copy(self):
        """Deep copy this lattice
        Returns:
            Lattice: copied lattice
        """
        return Lattice(self.basis.copy())

    def randomize(seed, scale=3, U=None):
        """Randomize this lattice basis and reset mu and B"""
        self.basis = randomized(self.basis, seed, scale=scale, U=U)
        self.set_gso_data()
        return self


def randomized(basis, seed, scale=3, U=None):
    """
    Args:
        basis (np.ndarray): lattice basis
        seed (int): seed for randomization
        scale (int): strength of randomization
        U (np.ndarray): inverse matrix of transformation
            such that UC = B, where B is an input basis and C is output basis
    Returns:
        np.ndarray: randomized basis
    """
    if seed is None:
        seed = random.random()
    # seed
    random.seed(seed)

    # randomized basis
    r_basis = np.array(basis, dtype=np.float64)

    n = basis.shape[0]
    indexes = list(range(n))

    # 1. permute rows
    n_iters = 4 * n
    for _ in range(n_iters):
        a, b = random.sample(indexes, 2)
        r_basis[[a, b]] = r_basis[[b, a]]
        if U is not None:
            U.T[[a, b]] = U.T[[b, a]]

    # 2. triangular transformation matrix with coefficients in -1, 0, 1
    flag = lambda: random.randint(0, 1)
    for a in range(n - 2):
        for _ in range(scale):
            b = random.randint(a + 1, n - 1)
            if flag():
                r_basis[a] += r_basis[b]
                if U is not None:
                    U.T[b] -= U.T[a]
            else:
                r_basis[a] -= r_basis[b]
                if U is not None:
                    U.T[b] += U.T[a]

    return r_basis
