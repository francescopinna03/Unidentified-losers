import numpy as np
from scipy.optimize import brentq, linprog
from scipy.special import xlogy, logsumexp

J = 25
W = np.array([0.40, 0.35, 0.25])
L = np.array([12000.0, 18000.0, 24000.0])
A = np.array([18000.0, 9000.0, 0.0])
D = np.array([2000.0, 1500.0, 1000.0])
KAPPA = 0.10
ALPHA = 0.10
MU = np.ones(J) / J
Y0 = [A[g] + L[g] * (np.arange(J) + 0.5) / J for g in range(3)]
Y1 = [A[g] + D[g] + L[g] * (np.arange(J) + 0.5) / J for g in range(3)]
REFERENCES = {
    "independent": np.ones((J, J)) / J**2,
    "persistent": (1 - KAPPA) * np.eye(J) / J + KAPPA * np.ones((J, J)) / J**2,
}


def loss_matrix(g, transfer):
    diff = Y0[g][:, None] - (Y1[g][None, :] + transfer)
    return (diff > 1e-9).astype(float)


def shortfall_matrix(g, transfer):
    diff = Y0[g][:, None] - (Y1[g][None, :] + transfer)
    return np.maximum(diff, 0.0) / 1000.0


def kl(p, r):
    return float(np.sum(xlogy(p, p / r)))


def sinkhorn(M, r, lam, tol=1e-11, iters=4000):
    span = M.max() - M.min()
    if span / lam < 600:
        k = r * np.exp((M - M.max()) / lam)
        v = np.ones(J)
        for _ in range(iters):
            u = MU / (k @ v)
            v = MU / (k.T @ u)
            p = u[:, None] * k * v
            if abs(p.sum(1) - MU).max() < tol:
                break
        return p
    logk = np.log(r) + M / lam
    lmu = np.log(MU)
    g = np.zeros(J)
    for _ in range(iters):
        f = lmu - logsumexp(logk + g[None, :], axis=1)
        g = lmu - logsumexp(logk + f[:, None], axis=0)
        p = np.exp(f[:, None] + logk + g[None, :])
        if abs(p.sum(1) - MU).max() < tol:
            break
    return p


def unrestricted_plan(M):
    n = J
    aeq = np.zeros((2 * n, n * n))
    for i in range(n):
        aeq[i, i * n:(i + 1) * n] = 1
    for j in range(n):
        aeq[n + j, j::n] = 1
    res = linprog(-M.ravel(), A_eq=aeq[:-1], b_eq=np.r_[MU, MU][:-1], bounds=(0, None), method="highs")
    return res.x.reshape(n, n)


def worst_case(matrices, reference, radius):
    r = REFERENCES[reference]
    if radius == 0:
        plans = [r] * 3
        return float(W @ [np.sum(M * p) for M, p in zip(matrices, plans)]), plans
    scale = max(1.0, max(M.max() - M.min() for M in matrices))

    def at(lam):
        plans = [sinkhorn(M, r, lam) for M in matrices]
        value = float(W @ [np.sum(M * p) for M, p in zip(matrices, plans)])
        budget = float(W @ [kl(p, r) for p in plans])
        return value, budget, plans

    lam_low = 0.012 * scale
    value, budget, plans = at(lam_low)
    if budget <= radius:
        plans = [unrestricted_plan(M) for M in matrices]
        return float(W @ [np.sum(M * p) for M, p in zip(matrices, plans)]), plans
    lam = brentq(lambda x: at(x)[1] - radius, lam_low, 5000.0 * scale, xtol=1e-8)
    value, budget, plans = at(lam)
    return value, plans


def unrestricted_value(matrices):
    plans = [unrestricted_plan(M) for M in matrices]
    return float(W @ [np.sum(M * p) for M, p in zip(matrices, plans)]), plans
