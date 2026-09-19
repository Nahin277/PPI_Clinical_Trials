"""Real-data applications: ACTG 175 (randomised trial) and NHANES (survey waves)."""
import numpy as np, pandas as pd, json, sys
sys.path.insert(0, ".")
import ppct
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

RNG = np.random.default_rng(20260918)
BASE = ["age", "wtkg", "hemo", "homo", "drugs", "karnof", "oprior", "z30",
        "preanti", "race", "gender", "str2", "symptom", "cd40", "cd80"]


# ==========================================================================
# ACTG 175
# ==========================================================================
def _read_rda(path, key):
    import pyreadr
    return pyreadr.read_r(path)[key]


def actg_setup(path="data/ACTG175.rda", hist_frac=0.50, seed=7):
    df = _read_rda(path, "ACTG175")
    rng = np.random.default_rng(seed)
    is_hist = rng.random(len(df)) < hist_frac
    hist, trial = df[is_hist].copy(), df[~is_hist].copy()
    # historical prognostic model: fit on ZDV-only ("untreated-like") patients
    h = hist[hist.treat == 0]
    model = GradientBoostingRegressor(n_estimators=300, max_depth=2,
                                      learning_rate=0.05, subsample=0.8,
                                      random_state=0)
    model.fit(h[BASE], h.cd420)
    trial["f"] = model.predict(trial[BASE])
    # honest out-of-sample R2, computed on the target trial's control arm
    c = trial[trial.treat == 0]
    r2_oos = r2_score(c.cd420, c.f)
    return df, hist, trial, model, r2_oos


def summarise(Y, T, F, label):
    Y = np.asarray(Y, float); T = np.asarray(T, int); F = np.asarray(F, float)
    res = ppct.run_all(Y, T, F)
    Yt, Yc = Y[T == 1], Y[T == 0]
    Ft, Fc = F[T == 1], F[T == 0]
    rho_t = np.corrcoef(Ft, Yt)[0, 1]
    rho_c = np.corrcoef(Fc, Yc)[0, 1]
    rows = []
    ref_se = res["Reference"].se
    for k, r in res.items():
        lo, hi = r.ci()
        rows.append({"dataset": label, "estimator": k, "ate": r.estimate,
                     "se": r.se, "ci_lo": lo, "ci_hi": hi, "p": r.pvalue(),
                     "lambda": r.lam, "se_ratio": r.se / ref_se,
                     "var_ratio": (r.se / ref_se) ** 2})
    meta = {"dataset": label, "m": int((T == 1).sum()), "n": int((T == 0).sum()),
            "rho_t": rho_t, "rho_c": rho_c,
            "sd_t": Yt.std(ddof=1), "sd_c": Yc.std(ddof=1), "sd_f": F.std(ddof=1)}
    return pd.DataFrame(rows), meta


def equal_power_table(meta, res_df, delta, alpha=0.05, power=0.80):
    """Total N needed for each estimator, indexed so Reference = 100."""
    m, n = meta["m"], meta["n"]
    N = m + n
    out = {}
    for _, r in res_df.iterrows():
        V = r["se"] ** 2 * N                     # asymptotic variance
        out[r["estimator"]] = ppct.sample_size(V, delta, alpha, power)
    ref = out["Reference"]
    return {k: {"N": v, "index": 100 * v / ref} for k, v in out.items()}


def shrink_control(data, ycol, tcol, fcol, m=200,
                   ns=(10, 20, 40, 80, 120, 200), reps=600, seed=1):
    """Resample a trial of m treated and n controls out of the observed data.

    Both arms are redrawn each replicate, so the spread across replicates is the
    sampling distribution of a design with those arm sizes.  Reported alongside
    the mean of each estimator's own analytic SE, which lets us check the
    closed-form variance against the empirical one.
    """
    rng = np.random.default_rng(seed)
    t = data[data[tcol] == 1].reset_index(drop=True)
    c = data[data[tcol] == 0].reset_index(drop=True)
    rows = []
    for n in ns:
        if n > len(c) or m > len(t):
            continue
        acc = {k: [] for k in ppct.ALL_ESTIMATORS}
        ses = {k: [] for k in ppct.ALL_ESTIMATORS}
        for _ in range(reps):
            ti = rng.choice(len(t), m, replace=False)
            ci = rng.choice(len(c), n, replace=False)
            sub = pd.concat([t.iloc[ti], c.iloc[ci]])
            for k, r in ppct.run_all(sub[ycol].values, sub[tcol].values,
                                     sub[fcol].values).items():
                acc[k].append(r.estimate)
                ses[k].append(r.se)
        for k in acc:
            rows.append({"n_control": n, "m_treated": m, "estimator": k,
                         "mean": np.mean(acc[k]),
                         "sd_empirical": np.std(acc[k], ddof=1),
                         "se_analytic": np.mean(ses[k])})
    df = pd.DataFrame(rows)
    ref = df[df.estimator == "Reference"].set_index("n_control")["sd_empirical"]
    df["sd_ratio_vs_reference"] = df.apply(
        lambda r: r.sd_empirical / ref[r.n_control], axis=1)
    return df


# ==========================================================================
# NHANES — the "survey wave" transfer-learning framing
# ==========================================================================
NH_NUM = ["Age", "BMI", "Weight", "Height", "Pulse", "TotChol", "Poverty",
          "DirectChol", "BPDiaAve"]
NH_CAT = ["Gender", "Race1", "Diabetes", "Smoke100"]


def nhanes_setup(path="data/NHANES.rda"):
    df = _read_rda(path, "NHANES")
    df = df[(df.Age >= 20) & (df.Age <= 79)]
    keep = NH_NUM + NH_CAT + ["BPSysAve", "PhysActive", "SurveyYr"]
    df = df[keep].dropna()
    df = df.drop_duplicates()
    X = pd.get_dummies(df[NH_NUM + NH_CAT], drop_first=True).astype(float)
    df = df.assign(**{c: X[c] for c in X.columns})
    feats = list(X.columns)
    train = df[df.SurveyYr == "2009_10"]
    target = df[df.SurveyYr == "2011_12"].copy()
    # prognostic model learnt from the EARLIER wave, inactive respondents only
    tr = train[train.PhysActive == "No"]
    model = make_pipeline(StandardScaler(),
                          GradientBoostingRegressor(n_estimators=400, max_depth=2,
                                                    learning_rate=0.05,
                                                    subsample=0.8, random_state=0))
    model.fit(tr[feats], tr.BPSysAve)
    target["f"] = model.predict(target[feats])
    target["T"] = (target.PhysActive == "Yes").astype(int)
    ctrl = target[target["T"] == 0]
    r2_oos = r2_score(ctrl.BPSysAve, ctrl.f)
    return train, target, model, feats, r2_oos


def matched_power_n(meta, res_df, target_sd=None):
    """Control-arm size a PPCT design needs to match a balanced reference RCT.

    Uses the paper's design rule n*/m = sqrt(1 - rho_c^2) (Eq. 7-8, Table 1).
    """
    return {"rho_c": meta["rho_c"],
            "ratio": float(ppct.control_arm_ratio(meta["rho_c"]))}


# ==========================================================================
if __name__ == "__main__":
    report = {}

    # ---------------- ACTG 175
    df, hist, trial, model, r2 = actg_setup()
    res, meta = summarise(trial.cd420.values, trial.treat.values,
                          trial.f.values, "ACTG175")
    meta["r2_oos_control"] = r2
    meta["n_historical"] = int(len(hist))
    meta["n_historical_control"] = int((hist.treat == 0).sum())
    meta["full_trial_N"] = int(len(df))
    res.to_csv("out_actg_estimators.csv", index=False)
    report["actg_meta"] = meta
    report["actg_equal_power"] = equal_power_table(
        meta, res, delta=res.loc[res.estimator == "Reference", "ate"].iloc[0])
    report["actg_ratio_theory"] = float(ppct.control_arm_ratio(meta["rho_c"]))

    ub = shrink_control(trial, "cd420", "treat", "f", m=200,
                        ns=(10, 20, 40, 80, 120, 200))
    ub.to_csv("out_actg_unbalanced.csv", index=False)

    # balance diagnostic: chance imbalance in the prognostic score
    fm = trial.groupby("treat").f.mean()
    report["actg_f_imbalance"] = float(fm[1] - fm[0])
    report["actg_shift_explained"] = float(
        res.loc[res.estimator == "PPCT", "lambda"].iloc[0] * (fm[1] - fm[0]))
    report["actg_shift_observed"] = float(
        res.loc[res.estimator == "PPCT", "ate"].iloc[0]
        - res.loc[res.estimator == "Reference", "ate"].iloc[0])

    # full-trial (no split) reference effect, for context
    full = df.copy()
    report["actg_full_ate"] = float(full[full.treat == 1].cd420.mean()
                                    - full[full.treat == 0].cd420.mean())

    # ---------------- NHANES
    tr, tgt, nmodel, feats, r2n = nhanes_setup()
    nres, nmeta = summarise(tgt.BPSysAve.values, tgt["T"].values,
                            tgt.f.values, "NHANES 2011-12")
    nmeta["r2_oos_control"] = r2n
    nmeta["n_train_wave"] = int(len(tr))
    nres.to_csv("out_nhanes_estimators.csv", index=False)
    report["nhanes_meta"] = nmeta
    report["nhanes_equal_power"] = equal_power_table(
        nmeta, nres, delta=nres.loc[nres.estimator == "Reference", "ate"].iloc[0])
    report["nhanes_ratio_theory"] = float(ppct.control_arm_ratio(nmeta["rho_c"]))

    nub = shrink_control(tgt, "BPSysAve", "T", "f", m=300,
                         ns=(25, 50, 100, 200, 300, 600))
    nub.to_csv("out_nhanes_unbalanced.csv", index=False)

    nfm = tgt.groupby("T").f.mean()
    report["nhanes_f_imbalance"] = float(nfm[1] - nfm[0])
    report["nhanes_shift_explained"] = float(
        nres.loc[nres.estimator == "PPCT", "lambda"].iloc[0] * (nfm[1] - nfm[0]))
    report["nhanes_shift_observed"] = float(
        nres.loc[nres.estimator == "PPCT", "ate"].iloc[0]
        - nres.loc[nres.estimator == "Reference", "ate"].iloc[0])
    report["nhanes_age_by_arm"] = tgt.groupby("T").Age.mean().to_dict()
    report["nhanes_bmi_by_arm"] = tgt.groupby("T").BMI.mean().to_dict()

    json.dump(report, open("out_real.json", "w"), indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))
    print("\nACTG estimators\n", res.to_string(index=False))
    print("\nNHANES estimators\n", nres.to_string(index=False))
