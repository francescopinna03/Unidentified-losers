import json
import numpy as np
from model import W, loss_matrix, worst_case

TAX = 0.30
ALLOWANCE = 2000.0
EFFECTIVE = ALLOWANCE / (1 - TAX)
RADIUS = 0.10


def main():
    losses = [loss_matrix(g, EFFECTIVE) for g in range(3)]
    out = {"radius": RADIUS}
    for reference in ["persistent", "independent"]:
        value, plans = worst_case(losses, reference, RADIUS)
        aggregate = [float(np.sum(M * p)) for M, p in zip(losses, plans)]
        separate = []
        for g in range(3):
            matrices = [losses[k] if k == g else np.zeros_like(losses[k]) for k in range(3)]
            cell_value, cell_plans = worst_case(matrices, reference, RADIUS)
            separate.append(float(np.sum(losses[g] * cell_plans[g])))
        out[reference] = {"aggregate_worst_case": value, "cells_in_aggregate_worst_case": aggregate,
                          "separate_cell_maxima": separate, "weighted_sum_of_separate_maxima": float(W @ separate)}
    json.dump(out, open("../results/incidence.json", "w"), indent=1)


if __name__ == "__main__":
    main()
