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
| `code/simulate.py` | Generator of the synthetic pilot |
| `code/model.py` | Earnings distributions, reference matchings and worst-case solver |
| `code/pilot.py` | Average effects by cell |
| `code/evaluation.py` | Shares of losers, breakdown points, shortfalls, MVPF and equal-outlay top-up |
| `code/welfare.py` | Distributional MVPF |
| `code/exhibits.py` | Exhibits 1 and 2 |
| `results/` | Values cited in the paper |

## Data

The data are generated from a population benchmark in which the program is evaluated within three cells defined by contract type, stable part-time employees, fixed-term workers and intermittent workers, with population weights 0.40, 0.35 and 0.25. In each cell, pre-tax earnings cumulated over the 24 months after assignment are uniform without the program, on intervals of width €12,000, €18,000 and €24,000 starting at €18,000, €9,000 and €0, so that untreated mean earnings are €24,000, €18,000 and €12,000, and the program translates the distribution by €2,000, €1,500 and €1,000. Earnings are taxed at 30%, while the allowance of €2,000 is paid tax-free to every assigned worker. The benchmark fixes the two marginal distributions in each cell and leaves their coupling unspecified, since the coupling of the two potential outcomes of the same worker is the object the paper studies and no lottery can reveal it.

The file `data/synthetic_pilot.csv` contains the synthetic lottery, with one row for each of 6,000 applicants, of whom 2,400, 2,100 and 1,500 belong to the three cells and exactly half of each cell is assigned to the offer.

| Variable | Description |
|---|---|
| `id` | Applicant identifier, from 1 to 6,000 |
| `group` | Cell, coded 1 for stable part-time employees, 2 for fixed-term workers and 3 for intermittent workers |
| `assignment` | 1 if the applicant was drawn for the offer, 0 otherwise |
| `income` | Pre-tax labor earnings over the 24 months after assignment, in euros, excluding the allowance |
| `pre_income` | Pre-tax labor earnings over the 24 months before assignment, in euros |

Within each cell the earnings of the two arms are drawn independently from the two marginal distributions of the benchmark, with seed 20260925, so that every applicant is observed in one arm only, exactly as in a real lottery. Pre-program earnings are drawn with seed 20260926, independently of the outcome and uniformly on a band of €6,000 around the cell mean, so they describe the stratification of the lottery and carry no information on the outcome. The script `code/simulate.py` regenerates the file byte for byte.

The file enters the computations through `code/pilot.py`, which estimates the average effect of the offer in each cell as a difference in means, with Wald intervals, and aggregates the cells with their population weights. The estimate of €1,529 is the gain used in the marginal value of public funds. The distributional computations, from the shares of losers to the shortfalls and the weighted marginal value of public funds, use the two marginal distributions of the benchmark, which the lottery identifies, with closed forms at the reference scenarios and without restrictions, and with a grid of 25 equiprobable midpoints per cell at a positive divergence from a scenario.

## Reproduction

The computations require Python 3.10 or later, with the packages listed in `requirements.txt`, and a LaTeX distribution that provides pdflatex, and the following commands rebuild the data, the results, the exhibits and the paper in about one minute.

```
pip install -r requirements.txt
cd code
python simulate.py
python pilot.py
python evaluation.py
python welfare.py
python exhibits.py
cd ../paper
pdflatex paper.tex
pdflatex paper.tex
```

Since the scripts are deterministic, they rewrite `data/` and `results/` with identical files, while the figures are drawn in Latin Modern when that font is installed and in another serif font otherwise.

## Provenance of the reported values

Every figure quoted in the paper is stored in one of the results files, under the key reported next to it in the following table.

| In the paper | File and key |
|---|---|
| Average effect of €1,529 and its interval, Exhibit 1A | `results/pilot.json` |
| After-tax average gain of €1,103 in the population | `results/evaluation.json`, `parameters` |
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

Version submitted on 29 October 2026 for the course Advanced Public Economics at LUISS Guido Carli.

## Citation

Pinna, F. (2026). *Evaluating a Retraining Program When Its Losers Are Unidentified*. Policy paper, LUISS Guido Carli.
