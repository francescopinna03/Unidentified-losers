import json
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid
from bridge import closed_form, SIGMA
from model import W, KAPPA

NODES = 4001


def cumulative(values, step):
    return np.concatenate([[0.0], cumulative_trapezoid(values, dx=step)])


class Residual:
    def __init__(self, rho, theta, sigma):
        self.x = np.linspace(0.0, 1.0, NODES)
        self.h = self.x[1] - self.x[0]
        self.rho = rho
        self.theta = theta
        self.c = np.expm1(theta)
        self.sigma = sigma
        mass = cumulative(rho, self.h)
        self.t = np.linspace(0.0, 1.0 - sigma, NODES)
        self.k = self.t[1] - self.t[0]
        self.rho_t = np.interp(self.t, self.x, rho)
        self.rho_s = np.interp(self.t + sigma, self.x, rho)
        self.drift = np.interp(self.t, self.x, mass) - np.interp(self.t + sigma, self.x, mass) + np.interp(sigma, self.x, mass)
        self.bottom = float(np.interp(sigma, self.x, mass))
        self.top = float(mass[-1] - np.interp(1.0 - sigma, self.x, mass))

    def logs(self, K):
        P = K + self.c * self.drift
        phi = cumulative(self.rho_t / P, self.k)
        psi = cumulative((self.rho_s / P)[::-1], self.k)[::-1]
        return self.c * phi, self.c * psi

    def closure(self, s):
        K = self.low + np.exp(s)
        logB, logA = self.logs(K)
        return K * ((self.c + 1.0) * np.exp(-logA[0]) - 1.0) / self.c - self.bottom

    def solve(self):
        self.low = max(0.0, -self.c * self.drift.min())
        s = brentq(self.closure, -60.0, 30.0, xtol=1e-13)
        K = self.low + np.exp(s)
        logB, logA = self.logs(K)
        self.K = K
        self.log_omega = np.log(K) - logA[0]
        self.logB = logB
        self.check = np.exp(self.log_omega) * (1.0 - np.expm1(logB[-1]) / self.c) - self.top
        self.log_ea = np.where(self.x > self.sigma, np.interp(self.x - self.sigma, self.t, logB), 0.0)
        self.log_eb = np.where(self.x <= 1.0 - self.sigma, np.interp(self.x, self.t, logA), 0.0)
        return self

    def loss_share(self):
        return float(np.exp(self.theta) / self.c * np.trapezoid(self.rho_s * (-np.expm1(-self.logB)), dx=self.k))


def solve_cell(theta, sigma, kappa, tol=1e-11, iters=4000):
    x = np.linspace(0.0, 1.0, NODES)
    rho = np.full(NODES, kappa)
    step = 1.0
    previous = np.inf
    for _ in range(iters):
        r = Residual(rho, theta, sigma).solve()
        q = (1.0 - kappa) / kappa * np.exp(-(r.log_omega + r.log_ea + r.log_eb))
        target = 2.0 / (1.0 + np.sqrt(1.0 + 4.0 * q))
        change = np.abs(target - rho).max()
        if change < tol:
            break
        if change > previous:
            step = max(step / 2.0, 1e-3)
        else:
            step = min(step * 1.1, 1.0)
        previous = change
        rho = np.exp((1.0 - step) * np.log(rho) + step * np.log(target))
    r = Residual(rho, theta, sigma).solve()
    H = r.loss_share()
    h = x[1] - x[0]
    KL = float(np.trapezoid(2 * np.log(rho) - r.log_ea - r.log_eb, dx=h) - r.log_omega - np.log(kappa) + theta * H)
    return {"H": H, "KL": KL, "diagonal_mass": float(1 - np.trapezoid(rho, dx=h)), "closure_check": float(r.check), "change": float(change)}


def aggregate(theta, kappa=KAPPA):
    cells = [solve_cell(theta, s, kappa) for s in SIGMA]
    return float(W @ [c["H"] for c in cells]), float(W @ [c["KL"] for c in cells]), cells


def scaled_lower_bound(delta, kappa=KAPPA):
    from bridge import aggregate as product_aggregate
    theta = brentq(lambda t: product_aggregate(t)[1] - delta / kappa, 1e-6, 200.0)
    return kappa * product_aggregate(theta)[0]


def donsker_varadhan_bound(delta, kappa=KAPPA):
    def value(theta):
        mgf = [1.0 + kappa * np.expm1(theta) * (1 - s) ** 2 / 2 for s in SIGMA]
        return (delta + float(W @ np.log(mgf))) / theta
    grid = np.linspace(0.01, 60, 6000)
    return float(min(value(t) for t in grid))


def main():
    out = {"kappa": KAPPA, "center": float(KAPPA * W @ [(1 - s) ** 2 / 2 for s in SIGMA]), "curve": []}
    for delta in [0.01, 0.02, 0.05, 0.1, 0.2]:
        theta = brentq(lambda t: aggregate(t)[1] - delta, 0.05, 15.0, xtol=1e-8)
        H, KL, cells = aggregate(theta)
        out["curve"].append({"radius": delta, "theta": theta, "aggregate": H, "cells": [c["H"] for c in cells],
                             "diagonal_mass": [c["diagonal_mass"] for c in cells],
                             "closure_check": max(abs(c["closure_check"]) for c in cells),
                             "lower_bound": scaled_lower_bound(delta), "upper_bound": donsker_varadhan_bound(delta)})
    out["breakdown_ten_percent"] = brentq(lambda d: aggregate(brentq(lambda t: aggregate(t)[1] - d, 0.05, 15.0, xtol=1e-8))[0] - 0.10, 0.05, 0.3, xtol=1e-6)
    json.dump(out, open("../results/persistent_bridge.json", "w"), indent=1)


if __name__ == "__main__":
    main()
