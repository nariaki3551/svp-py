import math
import argparse

import numpy as np

import svp.core
import svp.reduction
import svp.enumeration
import svp.optimization


algorithms = ["randomize", "lll", "enum", "iqp"]


def main(args):
    lattice = svp.core.Lattice.create_from_file(args.instance)

    for algo in args.algos:
        print("run", algo)
        vector = None

        # R = lattice.GH
        R = 1.01 * np.min(np.linalg.norm(lattice.basis, axis=1))

        if algo == "randomize":
            lattice.randomize(seed=0)
            vector = min(lattice.basis, key=lambda b: np.linalg.norm(b))
        if algo == "lll":
            svp.reduction.LLL(lattice, arg_params=args)
            vector = lattice.basis[0]
        if algo == "enum":
            if (coeffs := svp.enumeration.dfs(lattice, R, arg_params=args)) is not None:
                vector = coeffs.T.dot(lattice.basis)
        if algo == "iqp":
            if (coeffs := svp.optimization.iqp(lattice, R, arg_params=args)) is not None:
                vector = coeffs.T.dot(lattice.basis)

        print(algo, "is finished")
        if vector is not None:
            print("found", vector)
            print("norm", np.linalg.norm(vector))
        print("\n", " " * 5, "-" * 50, "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "instance",
        type=str,
        help="Instance file of SVP Challenge",
    )
    parser.add_argument(
        "algos",
        nargs="+",
        type=str,
        choices=algorithms,
        help=f"Select algorithms from {algorithms}",
    )
    main(parser.parse_args())
