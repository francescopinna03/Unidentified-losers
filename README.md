# Evaluating a Retraining Program When Its Losers Are Unidentified

Francesco Pinna, Department of Economics and Finance, LUISS Guido Carli

The policy paper *Evaluating a Retraining Program When Its Losers Are Unidentified* evaluates a hypothetical two-year retraining program, paired with a flat allowance of €2,000, through a synthetic randomized pilot of its offer, and asks what such a pilot identifies about the average earnings gain, the fiscal cost and the share of workers left below the earnings they would have had without the program. The code, the data and the results collected here reproduce every number and exhibit of the paper from synthetic data alone, so that no real individual and no administrative record is involved at any stage.

## Contents

The paper, its source and its two figures are in `paper/`, the synthetic lottery of 6,000 applicants is in `data/`, the scripts are in `code/`, and the values cited in the text are stored in `results/`, as the table below details file by file.

| Path | Content |
|---|---|
| `paper/paper.pdf`, `paper/paper.tex` | Paper and source |
| `paper/figures/` | Exhibits 1 and 2 |
| `data/synthetic_pilot.csv` | Synthetic lottery of 6,000 applicants in three cells |
| `code/model.py` | Earnings distributions, reference matchings and worst-case solver |
| `code/pilot.py` | Average effects by cell |
| `code/evaluation.py` | Shares of losers, breakdown points, shortfalls, MVPF and equal-outlay top-up |
| `code/welfare.py` | Distributional MVPF |
| `code/exhibits.py` | Exhibits 1 and 2 |
| `results/` | Values cited in the paper |

## Reproduction

The computations require Python 3.10 or later, with the packages listed in `requirements.txt`, and a LaTeX distribution that provides pdflatex, and the following commands rebuild the results, the exhibits and the paper in about one minute.

```
pip install -r requirements.txt
cd code
python pilot.py
python evaluation.py
python welfare.py
python exhibits.py
cd ../paper
pdflatex paper.tex
pdflatex paper.tex
```

Since the scripts are deterministic, they rewrite `results/` with the same values, while the figures are drawn in Latin Modern when that font is installed and in another serif font otherwise.

## Where the numbers come from

Every figure quoted in the paper is stored in one of the results files, under the key reported next to it in the following table.

| In the paper | File and key |
|---|---|
| Average effect of €1,529 and its interval, Exhibit 1A | `results/pilot.json` |
| MVPF at program costs of €2,000, €3,000 and €4,000, with its interval | `results/evaluation.json`, `mvpf` |
| Shares of losers of 2.6%, 25.9% and 71.3%, Exhibit 1B | `results/evaluation.json`, `continuous` |
| Breakdown at 5.8% and 22% of the gap and with other reshuffled shares | `results/evaluation.json`, `breakdown_*`, `kappa_sensitivity`, `reference_distance` |
| Worst-case curves and incidence by cell, Exhibit 2 | `results/evaluation.json`, `grid_curves` |
| Shortfalls of €85 and €852 at the references and of €3,280 per loser | `results/evaluation.json`, `continuous_shortfall_references` |
| Shortfalls of €396 and €1,310 at 5% of the gap | `results/evaluation.json`, `severity` |
| Worst shortfalls in Exhibit 3, from €2,474 to €660 | `results/evaluation.json`, `continuous_shortfall_unrestricted` |
| Top-up at the 67th percentile with a cap of 33.2% | `results/evaluation.json`, `equal_outlay_topup` |
| MVPF of the top-up under behavioral responses, with 56% of recipients, a fall of 0.015 per €100 and 0.56 | `results/evaluation.json`, `topup_behavioral` |
| Distributional MVPF, from 0.66 to 1.30 across matchings | `results/welfare.json` |

## Versions

Version 3.9, completed in October 2026 and submitted for the course Advanced Public Economics at LUISS Guido Carli.

## Citation

Pinna, F. (2026). *Evaluating a Retraining Program When Its Losers Are Unidentified*. Policy paper, LUISS Guido Carli.
