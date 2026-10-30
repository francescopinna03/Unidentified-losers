import csv
import numpy as np
from model import KAPPA

SEED = 20261024
BAND = 6000.0
MEANS = [24000.0, 18000.0, 12000.0]
SIZES = [2400, 2100, 1500]


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    worker = 0
    for g in range(3):
        n = SIZES[g]
        u1 = rng.uniform(size=n)
        reshuffle = rng.uniform(size=n) < KAPPA
        u2 = np.where(reshuffle, rng.uniform(size=n), u1)
        low = MEANS[g] - BAND / 2
        for a, b in zip(u1, u2):
            worker += 1
            rows.append([worker, g + 1, low + BAND * a, low + BAND * b])
    with open("../data/synthetic_panel.csv", "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "group", "income_two_years_before", "income_one_year_before"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
