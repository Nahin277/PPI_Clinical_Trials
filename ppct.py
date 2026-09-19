"""
Prediction-Powered Inference for Clinical Trials (PPCT) — estimator library.

Implements the estimators of
  Poulet, Tran, Tezenas du Montcel, Dubois, Durrleman & Jedynak (2025),
  "Prediction-powered inference for clinical trials: application to linear
   covariate adjustment", BMC Medical Research Methodology 25:204.

Notation (paper, Method section)
--------------------------------
  N       total number of trial participants
  m       number treated  (T = 1)
  n       number control  (T = 0),  N = n + m
  Y       observed continuous outcome
  f(X)    prognostic score from a model trained OUTSIDE this trial
  sigma_t, sigma_c   sd of Y in the treated / control arm
  sigma_f            sd of f(X) over the whole trial
  rho_t, rho_c       corr(f(X), Y) within the treated / control arm

Estimators
----------
  reference : difference in arm means
  ppi       : PPI of Angelopoulos et al. (lambda = 1)
  ppi_lam   : PPI++ for a user-supplied lambda
  ppct      : PPI++ at the variance-minimising lambda*, plugged in from the data
  ppct2     : PPCT with the pooled-covariance lambda* estimator (assumes rho_c = rho_t)
  ancova    : linear covariate adjustment on the prognostic score (arm-specific slopes)

All estimators centre f(X) at its trial-wide mean.  Centring is a no-op for the
PPI family (the lambda * f term enters both arms with the same coefficient and the
means cancel) but it is what makes the arm-specific-slope ANCOVA expression
unbiased, so it is applied uniformly.
"""

import numpy as np
from dataclasses import dataclass
from scipy import stats


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------
@dataclass
class ATEResult:
    name: str
    estimate: float
    variance: float          # variance OF THE ESTIMATOR (not the asymptotic V)
    lam: float = np.nan      # lambda actually used, where applicable

    @property
    def se(self):
        return np.sqrt(self.variance)

    def ci(self, alpha=0.05):
        z = stats.norm.ppf(1 - alpha / 2)
        return self.estimate - z * self.se, self.estimate + z * self.se

    def pvalue(self):
        if self.variance <= 0:
            return np.nan
        return 2 * (1 - stats.norm.cdf(abs(self.estimate) / self.se))

    def reject(self, alpha=0.05):
        lo, hi = self.ci(alpha)
        return not (lo <= 0 <= hi)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _split(Y, T, F):
    Y, T, F = np.asarray(Y, float), np.asarray(T, int), np.asarray(F, float)
    F = F - F.mean()                       # centre the prognostic score
    return Y[T == 1], Y[T == 0], F[T == 1], F[T == 0], F


def _moments(Yt, Yc, Ft, Fc, Fall):
    """Empirical sds and within-arm correlations (ddof=1 throughout)."""
    s_t = Yt.std(ddof=1)
    s_c = Yc.std(ddof=1)
    s_f = Fall.std(ddof=1)
    rho_t = np.corrcoef(Ft, Yt)[0, 1] if len(Yt) > 2 and Ft.std() > 0 else 0.0
    rho_c = np.corrcoef(Fc, Yc)[0, 1] if len(Yc) > 2 and Fc.std() > 0 else 0.0
    return s_t, s_c, s_f, np.nan_to_num(rho_t), np.nan_to_num(rho_c)


def lambda_star(s_t, s_c, s_f, rho_t, rho_c, m, n):
    """Variance-minimising lambda (Proposition 4).

    lambda* = (pi_c * sigma_t * rho_t + pi_t * sigma_c * rho_c) / sigma_f
    with pi_t = m/N, pi_c = n/N.
    """
    if s_f <= 0:
        return 0.0
    N = m + n
    pi_t, pi_c = m / N, n / N
    return (pi_c * s_t * rho_t + pi_t * s_c * rho_c) / s_f


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------
def reference(Y, T, F=None):
    """Difference in arm means.  Var = sigma_c^2/n + sigma_t^2/m."""
    Y, T = np.asarray(Y, float), np.asarray(T, int)
    Yt, Yc = Y[T == 1], Y[T == 0]
    m, n = len(Yt), len(Yc)
    est = Yt.mean() - Yc.mean()
    var = Yt.var(ddof=1) / m + Yc.var(ddof=1) / n
    return ATEResult("Reference", est, var, lam=0.0)


def ppi_lambda(Y, T, F, lam):
    """PPI++ estimator at a fixed lambda (Definition 4).

    ATE = mean_t(Y - lam*f) - mean_c(Y - lam*f)
    Var = (s_t^2 + lam^2 s_f^2 - 2 lam s_f s_t rho_t)/m
        + (s_c^2 + lam^2 s_f^2 - 2 lam s_f s_c rho_c)/n
    """
    Yt, Yc, Ft, Fc, Fall = _split(Y, T, F)
    m, n = len(Yt), len(Yc)
    s_t, s_c, s_f, rho_t, rho_c = _moments(Yt, Yc, Ft, Fc, Fall)
    est = (Yt - lam * Ft).mean() - (Yc - lam * Fc).mean()
    var = ((s_t**2 + lam**2 * s_f**2 - 2 * lam * s_f * s_t * rho_t) / m
           + (s_c**2 + lam**2 * s_f**2 - 2 * lam * s_f * s_c * rho_c) / n)
    return ATEResult(f"PPI(lambda={lam:.3f})", est, max(var, 0.0), lam=lam)


def ppi(Y, T, F):
    """Plain prediction-powered inference: lambda = 1 (Definition 3)."""
    r = ppi_lambda(Y, T, F, 1.0)
    return ATEResult("PPI", r.estimate, r.variance, lam=1.0)


def ppct(Y, T, F):
    """PPCT: PPI++ at the plug-in lambda* (Definition 5 / Proposition 4)."""
    Yt, Yc, Ft, Fc, Fall = _split(Y, T, F)
    m, n = len(Yt), len(Yc)
    s_t, s_c, s_f, rho_t, rho_c = _moments(Yt, Yc, Ft, Fc, Fall)
    lam = lambda_star(s_t, s_c, s_f, rho_t, rho_c, m, n)
    r = ppi_lambda(Y, T, F, lam)
    return ATEResult("PPCT", r.estimate, r.variance, lam=lam)


def ppct2(Y, T, F):
    """PPCT2: lambda* under the constant-treatment-effect assumption rho_c = rho_t.

    Then lambda* reduces to Cov(Y, f) / Var(f), estimated by pooling the
    within-arm centred products — far more stable when one arm is tiny.
    """
    Yt, Yc, Ft, Fc, Fall = _split(Y, T, F)
    m, n = len(Yt), len(Yc)
    dy = np.concatenate([Yt - Yt.mean(), Yc - Yc.mean()])
    df = np.concatenate([Ft - Ft.mean(), Fc - Fc.mean()])
    dof = m + n - 2
    s_f2 = Fall.var(ddof=1)
    lam = (dy @ df) / dof / s_f2 if (s_f2 > 0 and dof > 0) else 0.0
    r = ppi_lambda(Y, T, F, lam)
    return ATEResult("PPCT2", r.estimate, r.variance, lam=lam)


def ancova(Y, T, F):
    """Linear covariate adjustment on the prognostic score, arm-specific slopes.

    Fits Y ~ 1 + T + f~ + T:f~ with f~ centred, and reports the coefficient on T
    with a heteroskedasticity-robust (HC0 sandwich) variance.  Equivalent to
    Proposition 6's mean_t(Y - lam_t f) - mean_c(Y - lam_c f) with centred f.
    """
    Y = np.asarray(Y, float)
    T = np.asarray(T, float)
    F = np.asarray(F, float)
    Fc_ = F - F.mean()
    Z = np.column_stack([np.ones_like(Y), T, Fc_, T * Fc_])
    XtX_inv = np.linalg.pinv(Z.T @ Z)
    beta = XtX_inv @ Z.T @ Y
    resid = Y - Z @ beta
    meat = (Z * resid[:, None]).T @ (Z * resid[:, None])
    cov = XtX_inv @ meat @ XtX_inv          # HC0
    return ATEResult("ANCOVA", beta[1], cov[1, 1], lam=np.nan)


ALL_ESTIMATORS = {
    "Reference": reference,
    "PPI": ppi,
    "PPCT": ppct,
    "PPCT2": ppct2,
    "ANCOVA": ancova,
}


def run_all(Y, T, F):
    """Return {name: ATEResult} for every estimator."""
    return {k: fn(Y, T, F) for k, fn in ALL_ESTIMATORS.items()}


# --------------------------------------------------------------------------
# design quantities
# --------------------------------------------------------------------------
def sample_size(var_per_unit_N, delta, alpha=0.05, power=0.80):
    """Minimum N for a two-sided test (Proposition 8).

    N = V * ((z_{1-alpha/2} + z_{1-beta}) / delta)^2
    where V is the ASYMPTOTIC variance, i.e. Var(ATEhat) * N.
    """
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return var_per_unit_N * ((z_a + z_b) / delta) ** 2


def control_arm_ratio(rho_c):
    """n*/m = sqrt(1 - rho_c^2) — Table 1 of the paper.

    Size of the control arm, relative to a standard balanced RCT's arm size,
    needed for a PPCT-powered trial to match the balanced design's power.
    """
    rho_c = np.clip(np.asarray(rho_c, float), -1, 1)
    return np.sqrt(1 - rho_c**2)


def variance_theory(s_t, s_c, s_f, rho_t, rho_c, m, n):
    """Closed-form variances of Reference / PPI / PPCT for a given design."""
    lam = lambda_star(s_t, s_c, s_f, rho_t, rho_c, m, n)
    v_ref = s_c**2 / n + s_t**2 / m
    v_ppi = ((s_t**2 + s_f**2 - 2 * s_f * s_t * rho_t) / m
             + (s_c**2 + s_f**2 - 2 * s_f * s_c * rho_c) / n)
    v_ppct = v_ref - lam**2 * s_f**2 * (1 / n + 1 / m)
    return {"lambda_star": lam, "Reference": v_ref, "PPI": v_ppi, "PPCT": v_ppct}
