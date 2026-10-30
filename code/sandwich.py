import json
import numpy as np
from model import J, W, REFERENCES, loss_matrix, shortfall_matrix, worst_case


def additive_projection(C, R):
    n = J
    A = np.zeros((n * n, 2 * n))
    for i in range(n):
        for j in range(n):
            A[i * n + j, i] = 1.0
            A[i * n + j, n + j] = 1.0
    w = np.sqrt(R.ravel())
    coef = np.linalg.lstsq(A * w[:, None], C.ravel() * w, rcond=None)[0]
    return C - (coef[:n][:, None] + coef[n:][None, :])


def constants(matrices, R):
    perp = [additive_projection(M, R) for M in matrices]
    variance = float(W @ [np.sum(R * P**2) for P in perp])
    bound = max(float(np.abs(P).max()) for P in perp)
    return variance, bound


def main():
    allowance = 2000.0 / 0.7
    radii = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    out = {}
    for name, matrices in [("loss", [loss_matrix(g, allowance) for g in range(3)]),
                           ("shortfall", [shortfall_matrix(g, allowance) for g in range(3)])]:
        out[name] = {}
        for ref in ["independent", "persistent"]:
            R = REFERENCES[ref]
            variance, bound = constants(matrices, R)
            center = float(W @ [np.sum(R * M) for M in matrices])
            rows = []
            for radius in radii:
                exact = worst_case(matrices, ref, radius)[0] - center
                lead = np.sqrt(2 * radius * variance)
                rows.append({"radius": radius, "exact": exact, "lower": lead - radius * bound,
                             "leading": lead, "upper": lead + radius * bound / 3})
            out[name][ref] = {"center": center, "projected_variance": variance, "residual_bound": bound,
                              "informative_radius": 2 * variance / bound**2, "rows": rows}
    json.dump(out, open("../results/sandwich.json", "w"), indent=1)


if __name__ == "__main__":
    main()
