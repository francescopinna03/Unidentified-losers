import json
import numpy as np
from model import J, W, D, REFERENCES, Y0, Y1, loss_matrix, shortfall_matrix, worst_case, unrestricted_value, unrestricted_plan

ALLOWANCE = 2000.0
PROGRAM_COST = 3000.0
TAX = 0.30
EFFECTIVE = ALLOWANCE / (1 - TAX)
EFFECT = 1529.0
RADII = [0.0, 0.05, 0.10, 0.20]


def rank_weights(power=1):
    all_y0 = np.concatenate(Y0)
    all_w = np.concatenate([np.full(J, W[g] / J) for g in range(3)])
    order = np.argsort(all_y0)
    cdf = np.cumsum(all_w[order])
    f0 = np.empty_like(all_y0)
    f0[order] = cdf - all_w[order] / 2
    weights = (power + 1) * (1 - f0) ** power
    return [weights[g * J:(g + 1) * J] for g in range(3)]


def analysis(weights):
    net_cost = PROGRAM_COST + ALLOWANCE - TAX * EFFECT
    net_cost_program = PROGRAM_COST - TAX * EFFECT
    mean_weight = float(W @ [np.mean(w) for w in weights])
    weighted_y0 = float(W @ [np.mean(weights[g] * Y0[g]) for g in range(3)])
    matrices = [weights[g][:, None] * Y1[g][None, :] / 1000.0 for g in range(3)]

    def bounds(ref, radius):
        high, _ = worst_case(matrices, ref, radius)
        low, _ = worst_case([-M for M in matrices], ref, radius)
        return 1000 * (-low) - weighted_y0, 1000 * high - weighted_y0

    def dmvpf(gain, allowance, cost):
        return ((1 - TAX) * gain + allowance * mean_weight) / cost

    out = {"mean_weight": mean_weight, "net_cost": net_cost, "net_cost_program_only": net_cost_program}
    for ref in REFERENCES:
        out[ref] = {}
        for radius in RADII:
            low, high = bounds(ref, radius)
            out[ref][str(radius)] = {"weighted_gain": [low, high],
                                     "dmvpf_allowance": [dmvpf(low, ALLOWANCE, net_cost), dmvpf(high, ALLOWANCE, net_cost)],
                                     "dmvpf_program_only": [dmvpf(low, 0.0, net_cost_program), dmvpf(high, 0.0, net_cost_program)]}
    high, _ = unrestricted_value(matrices)
    low, _ = unrestricted_value([-M for M in matrices])
    low, high = 1000 * (-low) - weighted_y0, 1000 * high - weighted_y0
    out["unrestricted"] = {"weighted_gain": [low, high],
                           "dmvpf_allowance": [dmvpf(low, ALLOWANCE, net_cost), dmvpf(high, ALLOWANCE, net_cost)],
                           "dmvpf_program_only": [dmvpf(low, 0.0, net_cost_program), dmvpf(high, 0.0, net_cost_program)]}
    losses = [loss_matrix(g, EFFECTIVE) for g in range(3)]
    shortfalls = [(1 - TAX) * shortfall_matrix(g, EFFECTIVE) for g in range(3)]
    adversaries = {}
    for name, ref, radius in [("unrestricted", None, None), ("persistent_0.10", "persistent", 0.10), ("independent_0.10", "independent", 0.10)]:
        if ref is None:
            count_plans = [unrestricted_plan(M) for M in losses]
            money_plans = [unrestricted_plan(M) for M in shortfalls]
        else:
            count_plans = worst_case(losses, ref, radius)[1]
            money_plans = worst_case(shortfalls, ref, radius)[1]
        rows = {}
        for label, plans in [("count_adversary", count_plans), ("shortfall_adversary", money_plans)]:
            count = float(W @ [np.sum(M * p) for M, p in zip(losses, plans)])
            money = 1000 * float(W @ [np.sum(M * p) for M, p in zip(shortfalls, plans)])
            rows[label] = {"count": count, "shortfall": money, "per_loser": money / count}
        adversaries[name] = rows
    out["two_adversaries"] = adversaries
    return out


def main():
    out = analysis(rank_weights(1))
    out["steeper_weights"] = {k: v for k, v in analysis(rank_weights(2)).items() if k not in ("two_adversaries",)}
    json.dump(out, open("../results/welfare.json", "w"), indent=1)


if __name__ == "__main__":
    main()
