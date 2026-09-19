# Prediction-Powered Inference for Clinical Trials

**Missing data + transfer learning, and what it buys a randomized trial.**

A reproducible walkthrough of the prediction-powered estimators for the average treatment
effect (ATE) in a two-arm randomized trial: the **PPI** estimator of Angelopoulos et al.
(2023), the **PPI++** weighting of Angelopoulos, Duchi & Zrnic (2023), and the **PPCT**
estimator of Poulet et al. (2025) — benchmarked against the classical difference in means
and against ANCOVA on a prognostic score.

Companion code for the Medium post *"Your Control Arm Is a Missing-Data Problem."*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O1iMHHubitqRVcL1gQa8zscXnjm3UiDJ)

---

## The idea in one paragraph

Every randomized trial pays for a control arm in order to estimate a counterfactual it can
never observe — a missing-data problem. A machine learning model trained on an external
historical cohort can predict that counterfactual, but using it directly ("digital twins")
imports the model's bias straight into the treatment effect. Prediction-powered inference
uses the model *and* the control arm: the predictions enter the estimator twice with
opposite signs, so whatever the model gets systematically wrong cancels, and only its
*correlation* with the outcome survives — as variance reduction. The result is unbiased no
matter how wrong the model is, and strictly no worse than the classical estimator.

---

## The estimators

With `m` treated, `n` control, outcome `Y`, prognostic score `f(X)` from a model trained on
data disjoint from the trial, `σ_t, σ_c` the within-arm SDs, `σ_f` the SD of the score, and
`ρ_t, ρ_c` the within-arm correlations between `f(X)` and `Y`:

| Estimator | Definition | Variance |
|---|---|---|
| **Reference** | `mean_t(Y) − mean_c(Y)` | `σ²_t/m + σ²_c/n` |
| **PPI** | `mean_t(Y − f) − mean_c(Y − f)` | λ = 1 case below |
| **PPI++ / PPCT(λ)** | `mean_t(Y − λf) − mean_c(Y − λf)` | `(σ²_t + λ²σ²_f − 2λσ_fσ_tρ_t)/m + (σ²_c + λ²σ²_f − 2λσ_fσ_cρ_c)/n` |
| **PPCT** | PPI++ at `λ* = (π_c σ_t ρ_t + π_t σ_c ρ_c)/σ_f` | `σ²_t/m + σ²_c/n − (λ*)²σ²_f(1/n + 1/m)` |
| **PPCT2** | PPCT with `λ*` estimated under `ρ_c = ρ_t`, i.e. pooled `Cov(Y,f)/Var(f)` | as above |
| **ANCOVA** | OLS of `Y` on `[1, T, f̃, T·f̃]`, `f̃` centered; HC0 variance | sandwich |

Three properties do all the work:

1. **Unbiased for every λ.** `ATE_λ = (1−λ)·ATE_ref + λ·ATE_PPI`, a convex combination of
   two unbiased estimators. Model quality never enters the bias.
2. **Never worse than classical.** The subtracted term `(λ*)²σ²_f(1/n + 1/m)` is a square
   times a square. With a useless model `λ* → 0` and PPCT *is* the reference estimator.
3. **The exchange rate is ρ².** Variance falls to roughly `1 − ρ²` of the classical
   estimator's, and to match a balanced RCT's power the control arm can shrink to
   `n*/m = √(1 − ρ_c²)`.

All estimators center `f(X)` at its trial-wide mean. Centering is a no-op for the PPI family
(the `λf` terms cancel across arms) but is what makes the arm-specific-slope ANCOVA
expression unbiased, so it's applied uniformly.

---

## What's in the notebook

`PPI_clinical_trials.ipynb` runs end to end on a free Colab CPU runtime in ~5 minutes and
downloads its own data.

| Part | Contents |
|---|---|
| 1 | The classical estimator and its variance decomposition |
| 2 | One simulated trial, five estimators side by side |
| 3 | 4,000 Monte Carlo trials: unbiasedness and variance reduction |
| 4 | Replication of Table 2 of Poulet et al. — unbalanced arms, skewed outcomes |
| 5 | Misspecification stress test: a linear model in a non-linear world, mis-calibrated by +15 |
| 6 | Power curves, type-I error, and the sample-size design rule |
| 7 | **ACTG 175** — a real randomized HIV trial, 2,139 patients |
| 8 | **NHANES** — the same machinery without randomization, and why that breaks |

---

## Headline results

### Simulation — 4,000 trials, 100 treated / 100 controls, ρ = 0.9

| Estimator | Mean | SD | Variance vs Reference |
|---|---|---|---|
| Reference | 0.998 | 0.143 | 1.000 |
| PPI | 1.000 | 0.069 | 0.232 |
| PPCT | 1.000 | 0.063 | **0.193** |
| PPCT2 | 1.000 | 0.063 | 0.193 |
| ANCOVA | 1.000 | 0.063 | 0.192 |

True ATE = 1. Theory predicts `1 − ρ² = 0.190`; `λ*` averaged 0.809.

### Misspecification — true ATE = 2, model wrong in form *and* calibration

| Estimator | Mean | SD |
|---|---|---|
| Naive (ML only, no rectifier) | **−12.99** | 0.33 |
| Reference | 2.003 | 0.386 |
| PPI | 2.003 | 0.349 |
| PPCT / PPCT2 / ANCOVA | 2.002 | 0.348 |

The naive estimator is off by exactly the +15 calibration error, with the *smallest* SD in
the table. Every rectified estimator is unbiased.

### ACTG 175 — CD4 count at 20 weeks

Prognostic model: gradient boosting trained on 265 ZDV-only patients from a held-out
"historical" half of the trial. Target trial: 794 treated, 267 control.
Out-of-sample R² = 0.378, ρ_c = 0.62, ρ_t = 0.60.

| Estimator | ATE (cells/mm³) | SE | 95% CI | λ* | Var vs Ref |
|---|---|---|---|---|---|
| Reference | 45.59 | 9.80 | (26.40, 64.79) | — | 1.000 |
| PPI | 54.71 | 7.78 | (39.47, 69.96) | 1 | 0.631 |
| PPCT | 53.60 | 7.74 | (38.44, 68.76) | 0.878 | **0.624** |
| PPCT2 | 53.83 | 7.74 | (38.67, 69.00) | 0.903 | 0.624 |
| ANCOVA | 53.77 | 7.66 | (38.75, 68.79) | — | 0.612 |

An equally powered future trial needs **62%** of the participants. In resampled sub-trials,
the classical estimator with 200 controls is matched by PPCT with about **130**.

**Why the point estimate moved.** Randomization balances covariates in expectation, not in
any one trial. Here the treated arm drew a slightly worse-prognosis sample:

```
λ* × imbalance in f(X) = 0.878 × (−9.12) = −8.008
PPCT − Reference       = 53.60 − 45.59   = +8.008
```

The shift is exactly the correction for chance prognostic imbalance.

### NHANES — the cautionary case

Model trained on the 2009–10 wave; 2011–12 wave treated as the target study. Outcome:
systolic BP. "Treatment": self-reported physical activity. **Not randomized.**

| Estimator | Effect (mmHg) | 95% CI | p |
|---|---|---|---|
| Reference | **−2.84** | (−4.38, −1.29) | 0.0003 |
| PPI | +0.73 | (−0.53, 2.00) | 0.26 |
| PPCT | +0.66 | (−0.61, 1.92) | 0.31 |
| ANCOVA | +0.68 | (−0.61, 1.96) | 0.30 |

Same arithmetic as ACTG 175 (`λ* × imbalance = 3.49 = PPCT − Reference`), entirely different
meaning: the active respondents are 6.0 years younger and 1.9 BMI units lighter, and the
model knows it.

> **The variance reduction survives without randomization. The estimand does not.**
> `ATE_λ` targets the ATE *because* `E[f(X)|T=1] = E[f(X)|T=0]` under randomization. Break
> that and the same formula quietly estimates a prognostic-score-adjusted contrast — which
> is not identified, because it can only remove confounding the model's covariates capture.
> PPI fixes variance. It does not fix confounding.

---

## Data

Both datasets are public and are downloaded by the notebook at runtime.

| Dataset | What it is | Source |
|---|---|---|
| **ACTG 175** | Randomized trial, 2,139 HIV-positive adults, ZDV monotherapy vs. three alternative regimens (Hammer et al., *NEJM* 1996). CD4 at 20 weeks + 15 baseline covariates. | R package [`BART`](https://cran.r-project.org/package=BART) (also in [`speff2trial`](https://cran.r-project.org/package=speff2trial)) |
| **NHANES** | US health survey, 2009–10 and 2011–12 waves, 10,000 respondents. | R package [`NHANES`](https://cran.r-project.org/package=NHANES); original data from [CDC NCHS](https://www.cdc.gov/nchs/nhanes/) |

Both arrive as `.rda` files and are read with `pyreadr` — no R installation required.

---

## Repository layout

```
.
├── README.md
├── PPI_clinical_trials.ipynb   # the full analysis, executed, with outputs
├── ppct.py                     # estimator library (standalone, no dependencies beyond numpy/scipy)
├── sim.py                      # Monte Carlo experiments
├── misspec.py                  # misspecification stress test
├── real_data.py                # ACTG 175 and NHANES analyses
├── figs.py                     # figure generation
└── figures/                    # generated PNGs
```

### Running locally

```bash
pip install numpy pandas scipy scikit-learn matplotlib pyreadr
python sim.py          # Monte Carlo study      (~2 min)
python misspec.py      # misspecification test  (~30 s)
python real_data.py    # ACTG 175 + NHANES      (~3 min)
python figs.py         # all figures
```

Or just open the notebook in Colab — it is self-contained.

### Using the estimators on your own trial

```python
from ppct import run_all, sample_size, control_arm_ratio

# Y: outcomes, T: 0/1 treatment indicator, F: prognostic scores from an
#    externally-trained model (must not have seen these outcomes)
results = run_all(Y, T, F)

r = results["PPCT2"]          # PPCT2 is the recommended default
print(r.estimate, r.se, r.ci(), r.pvalue(), r.lam)

# design: total N for 80% power at effect size delta
N = sample_size(r.variance * len(Y), delta=5.0, alpha=0.05, power=0.80)

# design: controls per treated patient needed to match a balanced RCT
ratio = control_arm_ratio(rho_c=0.62)
```

**One hard requirement:** `F` must come from a model fitted on data *disjoint* from this
trial. If the model saw these outcomes, the correlations are optimistic and the variance
estimates are wrong.

---

## Practical notes

- **Prefer PPCT2 over PPCT.** The default `λ̂*` multiplies `ρ̂_c` by `m`, so with a small
  control arm a handful of patients can dominate it. PPCT2's pooled estimator was the most
  stable at every sample size tested, and had the best-calibrated type-I error (5.1% vs
  6.6% nominal 5% at m = 100, n = 20, ρ = 0.9).
- **ANCOVA breaks at tiny n.** PPCT and ANCOVA are asymptotically equivalent, but with
  n ≤ 5 controls ANCOVA's slope estimate falls apart (SD more than 10× the reference
  estimator's at n = 2) while the PPI form retains a closed-form variance. That gap is
  exactly the regime prediction-powered designs are built for.
- **Returns on model quality are non-linear.** The `1 − ρ²` curve is flat below ρ ≈ 0.6.
  A mediocre model buys almost nothing; check your out-of-sample R² before promising a
  smaller trial.
- **Predicting change is much harder than predicting level.** The prognostic score has to
  track individual variability over the trial window, not just the population mean
  trajectory. This is where Poulet et al.'s Alzheimer's application ran aground
  (R² ≈ 15% on a secondary endpoint, ≈ 0 on the primary imaging endpoint).
- **λ can be pre-specified.** Fixing λ from prior data preserves the exact finite-sample
  unbiasedness and a known variance, at the cost of some efficiency. Estimating it from the
  trial is more efficient but loosens the guarantee; the bias is empirically negligible.
- **For small trials, use a permutation test.** The normal approximation needs roughly
  n, m ≥ 20.

---

## References

**Primary source**

- Poulet P-E, Tran M, Tezenas du Montcel S, Dubois B, Durrleman S, Jedynak B (2025).
  Prediction-powered inference for clinical trials: application to linear covariate
  adjustment. *BMC Medical Research Methodology* 25:204.
  https://doi.org/10.1186/s12874-025-02647-6

**Prediction-powered inference**

- Angelopoulos AN, Bates S, Fannjiang C, Jordan MI, Zrnic T (2023). Prediction-powered
  inference. *Science* 382(6671):669–674. https://doi.org/10.1126/science.adi6000
- Angelopoulos AN, Duchi JC, Zrnic T (2023). PPI++: Efficient prediction-powered inference.
  https://arxiv.org/abs/2311.01453
- Schultz J, Prince JL, Jedynak BM (2024). Predictive powered inference for healthcare:
  relating optical coherence tomography scans to multiple sclerosis progression.
  *Machine Learning for Healthcare Conference*, PMLR.
  https://proceedings.mlr.press/v252/schultz24a.html

**Covariate adjustment and prognostic scores**

- Schuler A, Walsh D, Hall D, Walsh J, Fisher C (2022). Increasing the efficiency of
  randomized trial estimates via linear adjustment for a prognostic score.
  *International Journal of Biostatistics* 18(2):329–356.
  https://doi.org/10.1515/ijb-2021-0072
- Holzhauer B, Adewuyi ET (2023). "Super-covariates": using predicted control group outcome
  as a covariate in randomized clinical trials. *Pharmaceutical Statistics* 22(6):1062–1075.
  https://doi.org/10.1002/pst.2329
- Li L (2022). Prognostic factor analyses. In: Piantadosi S, Meinert CL (eds),
  *Principles and Practice of Clinical Trials*. Springer, 1771–1787.
  https://doi.org/10.1007/978-3-319-52636-2_121
- European Medicines Agency (2015). *Guideline on adjustment for baseline covariates in
  clinical trials.*
  https://www.ema.europa.eu/en/adjustment-baseline-covariates-clinical-trials-scientific-guideline

**External and historical controls**

- Signorovitch JE, Sikirica V, Erder MH, Xie J, Lu M, Hodgkins PS, et al. (2012).
  Matching-adjusted indirect comparisons: a new tool for timely comparative effectiveness
  research. *Value in Health* 15(6):940–947. https://doi.org/10.1016/j.jval.2012.05.004
- Burman CF, Hermansson E, Bock D, Franzén S, Svensson D (2024). Digital twins and Bayesian
  dynamic borrowing: two approaches for incorporating historical control data.
  *Pharmaceutical Statistics*. https://doi.org/10.1002/pst.2376

**Data**

- Hammer SM, Katzenstein DA, Hughes MD, et al. (1996). A trial comparing nucleoside
  monotherapy with combination therapy in HIV-infected adults with CD4 cell counts from 200
  to 500 per cubic millimeter. *New England Journal of Medicine* 335:1081–1090. https://www.nejm.org/doi/full/10.1056/NEJM199610103351501
- CDC National Center for Health Statistics. *National Health and Nutrition Examination
  Survey*, 2009–2010 and 2011–2012. https://www.cdc.gov/nchs/nhanes/

---

## License

Code released under the MIT License. The ACTG 175 and NHANES datasets remain under the terms
of their respective distributors.
