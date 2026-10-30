import json
import numpy as np
from scipy.optimize import brentq
from scipy.stats import spearmanr
from model import J, W, ALPHA, KAPPA, REFERENCES, persistent_reference, loss_matrix, worst_case, kl

TAX = 0.30
ALLOWANCE = 2000.0
EFFECTIVE = ALLOWANCE / (1 - TAX)
SMOOTHING = 0.02


def rank_bins(values):
    order = np.argsort(np.argsort(values))
    return (order * J) // len(values)


def empirical_copula(x, y):
    counts = np.zeros((J, J))
    for i, j in zip(rank_bins(x), rank_bins(y)):
        counts[i, j] += 1
    estimate = counts / counts.sum()
    return (1 - SMOOTHING) * estimate + SMOOTHING / J**2


def main():
    data = np.genfromtxt("../data/synthetic_panel.csv", delimiter=",", names=True)
    cells = []
    references = []
    for g in range(3):
        cell = data[data["group"] == g + 1]
        x = cell["income_two_years_before"]
        y = cell["income_one_year_before"]
        rho = float(spearmanr(x, y).correlation)
        estimate = empirical_copula(x, y)
        references.append(estimate)
        true = persistent_reference(KAPPA)
        cells.append({"group": g + 1, "n": int(len(cell)), "spearman": rho, "kappa_hat": 1 - rho,
                      "kl_true_to_estimate": kl(true, estimate), "kl_estimate_to_true": kl(estimate, true),
                      "expected_estimation_divergence": (J**2 - 1) / (2 * len(cell))})
    losses = [loss_matrix(g, EFFECTIVE) for g in range(3)]
    center = float(W @ [np.sum(M * R) for M, R in zip(losses, references)])
    curve = []
    for radius in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]:
        value, plans = worst_case(losses, references, radius)
        curve.append({"radius": radius, "aggregate": value, "cells": [float(np.sum(M * p)) for M, p in zip(losses, plans)]})
    breakdown = brentq(lambda x: worst_case(losses, references, x)[0] - ALPHA, 1e-4, 1.0, xtol=1e-6)
    declared = [c["aggregate"] for c in [{"aggregate": worst_case(losses, "persistent", r)[0]} for r in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]]]
    out = {"cells": cells, "center_weighted": center,
           "kappa_hat_weighted": float(W @ [c["kappa_hat"] for c in cells]),
           "kl_true_to_estimate_weighted": float(W @ [c["kl_true_to_estimate"] for c in cells]),
           "expected_estimation_divergence_weighted": float(W @ [c["expected_estimation_divergence"] for c in cells]),
           "curve": curve, "declared_reference_curve": declared, "breakdown_ten_percent": breakdown}
    json.dump(out, open("../results/mobility.json", "w"), indent=1)


if __name__ == "__main__":
    main()
