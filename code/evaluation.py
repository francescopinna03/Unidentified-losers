import json
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import dblquad
from model import W, L, A, D, ALPHA, KAPPA, REFERENCES, loss_matrix, shortfall_matrix, worst_case, unrestricted_value, persistent_reference, reference_distance

ALLOWANCE = 2000.0
TAX = 0.30
EFFECTIVE = ALLOWANCE / (1 - TAX)
PROGRAM_COSTS = [2000.0, 3000.0, 4000.0]
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
    out["continuous"] = {"no_allowance": continuous_shares(0.0), "allowance": continuous_shares(EFFECTIVE)}
    losses = [loss_matrix(g, EFFECTIVE) for g in range(3)]
    shortfalls = [(1 - TAX) * shortfall_matrix(g, EFFECTIVE) for g in range(3)]
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
    p = brentq(lambda q: (1 - TAX) * topup_cost(q) - ALLOWANCE, 0.2, 0.99)
    cells = [topup_center_independent(p, g) for g in range(3)]
    out["equal_outlay_topup"] = {"quantile": p, "dependence_free_cap": 1 - p, "center_independent_cells": cells,
                                 "center_independent_weighted": float(W @ cells),
                                 "center_persistent_weighted": float(KAPPA * (W @ cells))}

    def unrestricted_shortfall(threshold):
        s = (D + threshold) / L
        return float(W @ ((1 - TAX) * L * (1 - s) ** 2 / 4))

    out["continuous_shortfall_unrestricted"] = {"no_allowance": unrestricted_shortfall(0.0),
                                                "allowance": unrestricted_shortfall(EFFECTIVE),
                                                "topup": float(W @ ((1 - TAX) * L * (1 - p) ** 2 / 2)),
                                                "topup_bound_attainable": bool(np.all(p - D / L >= 1 - p))}

    def reference_shortfall(threshold):
        s = (D + threshold) / L
        return float(W @ ((1 - TAX) * L * (1 - s) ** 3 / 6))

    independent_shortfall = reference_shortfall(EFFECTIVE)
    independent_share = float(W @ ((1 - (D + EFFECTIVE) / L) ** 2 / 2))
    out["continuous_shortfall_references"] = {"independent": independent_shortfall,
                                              "persistent": KAPPA * independent_shortfall,
                                              "per_loser": independent_shortfall / independent_share}

    recipients = p - D / L
    recipient_earnings = A + D + (p * L - D) / 2
    base_mvpf, base_wtp, base_cost = mvpf(ALLOWANCE, 3000.0, TAX, EFFECT)
    responses = {}
    for reduction in [0.05, 0.10, 0.15]:
        forgone = float(W @ (recipients * reduction * recipient_earnings))
        responses[str(reduction)] = {"forgone_per_assigned": forgone, "mvpf": base_wtp / (base_cost + forgone)}
    out["topup_behavioral"] = {"recipient_share": float(W @ recipients), "mvpf_static": base_mvpf,
                                "mvpf_change_per_100": base_wtp / (base_cost + 100.0) - base_mvpf,
                                "proportional_reductions": responses}
    distance = reference_distance()
    out["reference_distance"] = {"independent_to_persistent": distance[0], "persistent_to_independent": distance[1],
                                 "breakdown_persistent_share": out["breakdown_persistent_10pct"] / distance[0],
                                 "breakdown_independent_50pct_share": out["breakdown_independent_50pct"] / distance[0]}
    kappas = {}
    for kappa in [0.05, 0.10, 0.20]:
        r = persistent_reference(kappa)
        center = worst_case(losses, r, 0.0)[0]
        if center < ALPHA:
            breakdown = brentq(lambda x: worst_case(losses, r, x)[0] - ALPHA, 1e-4, 1.0, xtol=1e-6)
        else:
            breakdown = 0.0
        kappas[str(kappa)] = {"center": center, "breakdown_10pct": breakdown,
                              "breakdown_share_of_distance": breakdown / reference_distance(kappa)[0]}
    independent_continuous = out["continuous"]["allowance"]["independent"]["weighted"]
    for key in kappas:
        kappas[key]["center_continuous"] = float(key) * independent_continuous
    out["kappa_sensitivity"] = kappas
    out["parameters"]["after_tax_gain_population"] = (1 - TAX) * EFFECT_POPULATION
    json.dump(out, open("../results/evaluation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
