import csv
import numpy as np
from model import W, L, D, A

ORIGIN = np.array([8000.0, 10000.0, 12000.0])


def main():
    rng = np.random.default_rng(20260925)
    sizes = np.rint(6000 * W / 2).astype(int)
    rows = []
    for g in range(3):
        y0 = ORIGIN[g] + L[g] * rng.random(sizes[g])
        y1 = ORIGIN[g] + D[g] + L[g] * rng.random(sizes[g])
        for arm, y in [(0, y0), (1, y1)]:
            for value in y:
                rows.append([len(rows) + 1, g + 1, arm, float(value)])
    pre = np.random.default_rng(20260926)
    out = []
    for i, g, arm, value in rows:
        k = g - 1
        out.append([i, g, arm, float(value + A[k] - ORIGIN[k]),
                    float(pre.uniform(A[k] + L[k] / 2 - 3000, A[k] + L[k] / 2 + 3000))])
    with open("../data/synthetic_pilot.csv", "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "group", "assignment", "income", "pre_income"])
        writer.writerows(out)


if __name__ == "__main__":
    main()
