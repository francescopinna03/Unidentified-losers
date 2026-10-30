import json
import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw
from model import W, L, D, ALPHA

TAX = 0.30
ALLOWANCE = 2000.0
SIGMA = (D + ALLOWANCE / (1 - TAX)) / L


def closed_form(theta, sigma):
    c = np.expm1(theta)
    z = lambertw((1 - sigma) / sigma * np.exp(theta + (1 - sigma) / sigma)).real
    E = np.exp(z - (1 - sigma) / sigma)
    W_ = sigma * c / (np.exp(theta) - E)
    H = (np.exp(theta) / c) * ((1 - sigma) - sigma * (E - 1) / (np.exp(theta) - E))
    KL = -np.log(W_) - (1 - sigma) * np.log(E) + theta * H
    return float(H), float(KL), float(E), float(W_)


def projected_variance(sigma):
    s = 1 - sigma
    return s**2 / 2 - 2 * s**3 / 3 + s**4 / 4


def aggregate(theta):
    rows = [closed_form(theta, s) for s in SIGMA]
    return float(W @ [r[0] for r in rows]), float(W @ [r[1] for r in rows]), [r[0] for r in rows]


def main():
    out = {"sigma": SIGMA.tolist(), "center": [float((1 - s)**2 / 2) for s in SIGMA],
           "center_weighted": float(W @ [(1 - s)**2 / 2 for s in SIGMA]),
           "projected_variance": [projected_variance(s) for s in SIGMA],
           "projected_variance_weighted": float(W @ [projected_variance(s) for s in SIGMA])}
    curve = []
    for delta in [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]:
        theta = brentq(lambda t: aggregate(t)[1] - delta, 1e-4, 80.0)
        H, KL, cells = aggregate(theta)
        curve.append({"radius": delta, "theta": theta, "aggregate": H, "cells": cells})
    out["curve"] = curve
    breakdowns = {}
    for name, alpha in [("half", 0.5), ("ten_percent", ALPHA)]:
        if aggregate(1e-3)[0] < alpha:
            theta = brentq(lambda t: aggregate(t)[0] - alpha, 1e-4, 80.0)
            breakdowns[name] = {"theta": theta, "radius": aggregate(theta)[1]}
        else:
            breakdowns[name] = {"theta": 0.0, "radius": 0.0}
    out["breakdown_independent"] = breakdowns
    out["local_law_check"] = []
    s = float(SIGMA[0])
    H0 = (1 - s)**2 / 2
    for theta in [0.02, 0.05, 0.1, 0.2]:
        H, KL, _, _ = closed_form(theta, s)
        out["local_law_check"].append({"theta": theta, "radius": KL, "gain": H - H0,
                                       "leading": float(np.sqrt(2 * KL * projected_variance(s)))})
    json.dump(out, open("../results/bridge.json", "w"), indent=1)


if __name__ == "__main__":
    main()
