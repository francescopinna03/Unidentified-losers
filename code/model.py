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
def persistent_reference(kappa):
    return (1 - kappa) * np.eye(J) / J + kappa * np.ones((J, J)) / J**2


REFERENCES = {
    "independent": np.ones((J, J)) / J**2,
    "persistent": persistent_reference(KAPPA),
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
    r = REFERENCES[reference] if isinstance(reference, str) else reference
    if radius == 0:
        plans = [r] * 3
        return float(W @ [np.sum(M * p) for M, p in zip(matrices, plans)]), plans
    scale = max(1.0, max(M.max() - M.min() for M in matrices))

    def at(lam, iters=4000):
        plans = [sinkhorn(M, r, lam, iters=iters) for M in matrices]
        value = float(W @ [np.sum(M * p) for M, p in zip(matrices, plans)])
        budget = float(W @ [kl(p, r) for p in plans])
        return value, budget, plans

    lam_low = 0.012 * scale
    value, budget, plans = at(lam_low)
    if budget <= radius:
        free = [unrestricted_plan(M) for M in matrices]
        free_value = float(W @ [np.sum(M * p) for M, p in zip(matrices, free)])
        free_budget = float(W @ [kl(p, r) for p in free])
        if free_budget <= radius:
            return free_value, free
        lam_high = lam_low
        inner = (value, budget, plans)
        lam_try = lam_low
        for _ in range(16):
            lam_try = 0.7 * lam_try
            value, budget, plans = at(lam_try, 40000)
            if max(abs(p.sum(1) - MU).max() for p in plans) > 1e-9:
                break
            if budget > radius:
                lam = brentq(lambda x: at(x, 40000)[1] - radius, lam_try, lam_high, xtol=1e-8)
                value, budget, plans = at(lam, 40000)
                return value, plans
            lam_high = lam_try
            inner = (value, budget, plans)
        value, budget, plans = inner
        t = (radius - budget) / (free_budget - budget)
        plans = [(1 - t) * p + t * q for p, q in zip(plans, free)]
        return (1 - t) * value + t * free_value, plans
    lam = brentq(lambda x: at(x)[1] - radius, lam_low, 5000.0 * scale, xtol=1e-8)
    value, budget, plans = at(lam)
    return value, plans


def unrestricted_value(matrices):
    plans = [unrestricted_plan(M) for M in matrices]
    return float(W @ [np.sum(M * p) for M, p in zip(matrices, plans)]), plans


def reference_distance(kappa=KAPPA):
    r = persistent_reference(kappa)
    return kl(REFERENCES["independent"], r), kl(r, REFERENCES["independent"])
