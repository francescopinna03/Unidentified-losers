import json
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import dblquad
from model import W, L, D, ALPHA, KAPPA, REFERENCES, loss_matrix, shortfall_matrix, worst_case, unrestricted_value

ALLOWANCE = 2000.0
PROGRAM_COSTS = [2000.0, 3000.0, 4000.0]
TAX = 0.30
EFFECT = 1529.0
EFFECT_LOW = 1270.0
EFFECT_HIGH = 1789.0
EFFECT_POPULATION = 1575.0
RADII = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]


def continuous_shares(transfer):
    independent = [(1 - (D[g] + transfer) / L[g]) ** 2 / 2 for g in range(3)]
    persistent = [KAPPA * x for x in independent]
    unrestricted = [1 - (D[g] + transfer) / L[g] for g in range(3)]
    return {
        "independent": {"cells": independent, "weighted": float(W @ independent)},
        "persistent": {"cells": persistent, "weighted": float(W @ persistent)},
        "unrestricted": {"cells": unrestricted, "weighted": float(W @ unrestricted)},
    }


def mvpf(allowance, program_cost, tax, effect):
    wtp = (1 - tax) * effect + allowance
    cost = program_cost + allowance - tax * effect
    return wtp / cost, wtp, cost


def topup_cost(p):
    return float(W @ [max(p * L[g] - D[g], 0) ** 2 / (2 * L[g]) for g in range(3)])


def topup_center_independent(p, g):
    d = D[g] / L[g]
    return float(dblquad(lambda v, u: 1.0, p, 1.0, lambda u: 0.0, lambda u: max(min(u - d, 1.0), 0.0))[0])


def main():
    out = {"parameters": {"allowance": ALLOWANCE, "program_costs": PROGRAM_COSTS, "tax": TAX, "effect": EFFECT,
                          "effect_interval": [EFFECT_LOW, EFFECT_HIGH], "effect_population": EFFECT_POPULATION}}
    out["continuous"] = {"no_allowance": continuous_shares(0.0), "allowance": continuous_shares(ALLOWANCE)}
    losses = [loss_matrix(g, ALLOWANCE) for g in range(3)]
    shortfalls = [shortfall_matrix(g, ALLOWANCE) for g in range(3)]
    curves = {}
    for ref in REFERENCES:
        curves[ref] = []
        for radius in RADII:
            value, plans = worst_case(losses, ref, radius)
            cells = [float(np.sum(M * p)) for M, p in zip(losses, plans)]
            curves[ref].append({"radius": radius, "aggregate": value, "cells": cells})
    out["grid_curves"] = curves
    value, plans = unrestricted_value(losses)
    out["grid_unrestricted"] = {"weighted": value, "cells": [float(np.sum(M * p)) for M, p in zip(losses, plans)]}
    out["breakdown_persistent_10pct"] = brentq(lambda x: worst_case(losses, "persistent", x)[0] - ALPHA, 1e-4, 1.0, xtol=1e-6)
    out["breakdown_independent_50pct"] = brentq(lambda x: worst_case(losses, "independent", x)[0] - 0.50, 1e-4, 2.0, xtol=1e-6)
    severity = {}
    for ref in REFERENCES:
        v0, _ = worst_case(shortfalls, ref, 0.0)
        v1, p1 = worst_case(shortfalls, ref, 0.10)
        c0, _ = worst_case(losses, ref, 0.0)
        severity[ref] = {"center": 1000 * v0, "radius_0.10": 1000 * v1,
                         "cells_0.10": [1000 * float(np.sum(M * p)) for M, p in zip(shortfalls, p1)],
                         "per_loser_center": 1000 * v0 / c0}
    severity["unrestricted"] = 1000 * unrestricted_value(shortfalls)[0]
    out["severity"] = severity
    table = []
    for allowance in [0.0, ALLOWANCE]:
        for cost in PROGRAM_COSTS:
            m, wtp, net = mvpf(allowance, cost, TAX, EFFECT)
            table.append({"allowance": allowance, "program_cost": cost, "wtp": wtp, "net_cost": net, "mvpf": m,
                          "mvpf_low": mvpf(allowance, cost, TAX, EFFECT_LOW)[0],
                          "mvpf_high": mvpf(allowance, cost, TAX, EFFECT_HIGH)[0]})
    out["mvpf"] = table
    p = brentq(lambda q: topup_cost(q) - ALLOWANCE, 0.2, 0.99)
    cells = [topup_center_independent(p, g) for g in range(3)]
    out["equal_outlay_topup"] = {"quantile": p, "dependence_free_cap": 1 - p, "center_independent_cells": cells,
                                 "center_independent_weighted": float(W @ cells),
                                 "center_persistent_weighted": float(KAPPA * (W @ cells))}
    json.dump(out, open("../results/evaluation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
