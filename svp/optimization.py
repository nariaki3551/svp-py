import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def iqp(lattice, R, arg_params=None):
    """Integer quadratic programming of enumeration algorithm

    Args:
        lattice (Lattice) : lattice
        R (float) : upper bound of projected vector.
            sqnorm(pi_l(v)) <= R^2 for any l, 0 <= l < lattice.dim
        arg_params (Namespace): parameters from argument parser
    """
    n = lattice.dim
    basis = lattice.basis
    mu = lattice.mu
    B = lattice.B

    # create model
    model = gp.Model("enumeration_iqp")

    # calculate lower and upper bound of variables
    basis_tilde = np.linalg.inv(basis.dot(basis.T)).dot(basis)
    lb = -np.ceil(R * np.linalg.norm(basis_tilde, axis=1))
    ub = np.floor(R * np.linalg.norm(basis_tilde, axis=1))

    # generate variables
    v = model.addVars(n, vtype=GRB.CONTINUOUS, name=f"v", lb=lb, ub=ub)
    # v = model.addVars(n, vtype=GRB.INTEGER, name=f"v", lb=lb, ub=ub)
    norm = model.addVar(vtype=GRB.CONTINUOUS, name=f"norm", lb=0.0)

    # set objective : bb[i] = i-th element of sum(v_i b_i)
    bb = model.addVars(n, vtype=GRB.CONTINUOUS, name=f"bb", lb=-float("inf"))
    for j in range(n):
        model.addConstr(bb[j] == gp.quicksum(v[i] * basis[i, j] for i in range(n)))
    model.addGenConstrNorm(norm, bb, 2.0)
    model.setObjective(norm)

    # add constraints for preventing norm becomes 0 vector
    zero_ys = list()
    for i in range(n):
        ny = int(ub[i] - lb[i] + 1)
        y = model.addVars(ny, vtype=GRB.BINARY, name=f"y_{i}_")
        model.addConstr(v[i] == gp.quicksum((lb[i] + j) * y[j] for j in range(ny)))
        model.addConstr(gp.quicksum(y) == 1)
        zero_ys.append(y[-lb[i]])
    model.addConstr(gp.quicksum(zero_ys) <= n - 1)

    # add constraints for norm of projected vectors : sqnorm(pi_l(v)) <= R^2
    projected_sqnorm = 0.0
    for l in range(n - 1, -1, -1):
        p = gp.quicksum(mu[i, l] * v[i] for i in range(l, n))
        projected_sqnorm += p * p * B[l]
        model.addConstr(projected_sqnorm <= R**2)

    # set parameters
    model.setParam(GRB.Param.TuneOutput, 3)
    model.setParam(GRB.Param.Threads, 1)

    # solve problem
    model.optimize()

    # get status
    status = model.status
    if status == GRB.OPTIMAL:
        # fetch solution
        solution = model.getAttr("X", v)
        best_v = np.array([round(solution[i]) for i in range(n)])
        print("solution", solution)
        print("vector", best_v.T.dot(basis))
        return best_v
    elif status == GRB.INFEASIBLE:
        return None
    else:
        raise NotImplementedError
