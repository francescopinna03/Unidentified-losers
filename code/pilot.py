import json
import numpy as np
from model import W


def main():
    data = np.genfromtxt("../data/synthetic_pilot.csv", delimiter=",", names=True)
    groups = []
    for g in range(3):
        cell = data[data["group"] == g + 1]
        control = cell["income"][cell["assignment"] == 0]
        treated = cell["income"][cell["assignment"] == 1]
        effect = float(treated.mean() - control.mean())
        se = float(np.sqrt(control.var(ddof=1) / len(control) + treated.var(ddof=1) / len(treated)))
        groups.append({"group": g + 1, "n_per_arm": int(len(control)), "mean_control": float(control.mean()),
                       "mean_treated": float(treated.mean()), "effect": effect, "se": se,
                       "ci_low": effect - 1.96 * se, "ci_high": effect + 1.96 * se})
    effect = float(W @ [z["effect"] for z in groups])
    se = float(np.sqrt((W**2) @ [z["se"] ** 2 for z in groups]))
    out = {"n": int(len(data)), "groups": groups,
           "overall": {"effect": effect, "se": se, "ci_low": effect - 1.96 * se, "ci_high": effect + 1.96 * se}}
    json.dump(out, open("../results/pilot.json", "w"), indent=1)


if __name__ == "__main__":
    main()
