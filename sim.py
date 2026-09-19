"""Monte-Carlo study for the PPCT post."""
import numpy as np, pandas as pd, json, sys
sys.path.insert(0, ".")
import ppct

RNG = np.random.default_rng(20260918)


# ---------------------------------------------------------------- generators
def gen_trial(m, n, rho, delta=1.0, dist="normal", rng=RNG):
    """Paper's simulation model, eqs (13)-(16).

    Y_c ~ F,  Y_t = Y_c + delta,  f(X) = Y_c + eps,
    eps ~ N(0, sigma_eps^2) with sigma_eps^2 = sigma_c^2 (1-rho^2)/rho^2
    so that corr(f, Y) = rho in both arms.
    """
    N = m + n
    if dist == "normal":
        Yc0 = rng.normal(0, 1, N)
        s_c = 1.0
    else:                                   # chi-square with 5 df
        Yc0 = rng.chisquare(5, N)
        s_c = np.sqrt(10.0)
    T = np.r_[np.ones(m, int), np.zeros(n, int)]
    Y = Yc0 + delta * T
    if rho <= 0:
        F = rng.normal(0, s_c, N)           # uncorrelated predictor
    else:
        s_eps = s_c * np.sqrt((1 - rho**2) / rho**2)
        F = Yc0 + rng.normal(0, s_eps, N)
    return Y, T, F


def gen_nonlinear(N, rng=RNG, p=5, delta=2.0, misspec=True):
    """A misspecified-prognostic-model world.

    The truth is non-linear in X; the 'historical' model is a plain linear fit,
    so f(X) is systematically wrong — and PPI/PPCT must still be unbiased.
    """
    X = rng.normal(0, 1, (N, p))
    mu = 3 * np.sin(1.5 * X[:, 0]) + 2 * X[:, 1] ** 2 - 1.5 * X[:, 2] * X[:, 3] + X[:, 4]
    T = rng.binomial(1, 0.5, N)
    Y = mu + delta * T + rng.normal(0, 2.0, N)
    if misspec:
        f = 1.0 + 0.8 * X[:, 0] + 0.5 * X[:, 1]        # crude linear, biased
    else:
        f = mu
    return X, Y, T, f


# ---------------------------------------------------------------- experiments
def table2(reps=2000):
    """Reproduce the structure of the paper's Table 2."""
    rows = []
    for dist in ["normal", "chisq"]:
        for rho in [0.0, 0.5, 0.95]:
            for n in [2, 5, 10, 50, 100]:
                est = {k: [] for k in ppct.ALL_ESTIMATORS}
                for _ in range(reps):
                    Y, T, F = gen_trial(100, n, rho, 1.0, dist)
                    for k, r in ppct.run_all(Y, T, F).items():
                        est[k].append(r.estimate)
                row = {"dist": dist, "rho": rho, "n": n}
                for k, v in est.items():
                    v = np.asarray(v)
                    row[f"{k}_mean"] = v.mean()
                    row[f"{k}_sd"] = v.std(ddof=1)
                rows.append(row)
    return pd.DataFrame(rows)


def sampling_dists(reps=4000, m=100, n=100, rho=0.9):
    out = {k: [] for k in ppct.ALL_ESTIMATORS}
    lams = []
    for _ in range(reps):
        Y, T, F = gen_trial(m, n, rho, 1.0, "normal")
        r = ppct.run_all(Y, T, F)
        for k, v in r.items():
            out[k].append(v.estimate)
        lams.append(r["PPCT"].lam)
    return pd.DataFrame(out), np.array(lams)


def unbalanced(reps=2000, m=100, rho=0.9):
    ns = [2, 5, 10, 20, 50, 100]
    rows = []
    for n in ns:
        d = {k: [] for k in ppct.ALL_ESTIMATORS}
        for _ in range(reps):
            Y, T, F = gen_trial(m, n, rho, 1.0, "normal")
            for k, v in ppct.run_all(Y, T, F).items():
                d[k].append(v.estimate)
        for k, v in d.items():
            rows.append({"n": n, "estimator": k,
                         "mean": np.mean(v), "sd": np.std(v, ddof=1)})
    return pd.DataFrame(rows)


def misspecification(reps=2000, N=400):
    rows = []
    for _ in range(reps):
        _, Y, T, f = gen_nonlinear(N, misspec=True)
        r = ppct.run_all(Y, T, f)
        rows.append({k: v.estimate for k, v in r.items()}
                    | {"naive_f": f.mean(), "lam": r["PPCT"].lam})
    return pd.DataFrame(rows)


def power_curve(reps=2000, m=100, n=100, delta=0.3):
    """Power vs prediction quality (rho)."""
    rhos = np.linspace(0.0, 0.95, 12)
    rows = []
    for rho in rhos:
        hit = {k: 0 for k in ppct.ALL_ESTIMATORS}
        for _ in range(reps):
            Y, T, F = gen_trial(m, n, rho, delta, "normal")
            for k, v in ppct.run_all(Y, T, F).items():
                hit[k] += int(v.reject(0.05))
        for k, v in hit.items():
            rows.append({"rho": rho, "estimator": k, "power": v / reps})
    return pd.DataFrame(rows)


def type1(reps=4000, m=100, n=20, rho=0.9):
    hit = {k: 0 for k in ppct.ALL_ESTIMATORS}
    for _ in range(reps):
        Y, T, F = gen_trial(m, n, rho, 0.0, "normal")
        for k, v in ppct.run_all(Y, T, F).items():
            hit[k] += int(v.reject(0.05))
    return {k: v / reps for k, v in hit.items()}


if __name__ == "__main__":
    import time
    t0 = time.time()
    out = {}

    print("table2 ...", flush=True)
    t2 = table2(1500); t2.to_csv("out_table2.csv", index=False)

    print("sampling dists ...", flush=True)
    sd, lams = sampling_dists()
    sd.to_csv("out_sampling.csv", index=False)
    np.save("out_lams.npy", lams)

    print("unbalanced ...", flush=True)
    ub = unbalanced(); ub.to_csv("out_unbalanced.csv", index=False)

    print("misspecification ...", flush=True)
    ms = misspecification(); ms.to_csv("out_misspec.csv", index=False)

    print("power ...", flush=True)
    pw = power_curve(); pw.to_csv("out_power.csv", index=False)

    print("type1 ...", flush=True)
    out["type1"] = type1()

    json.dump(out, open("out_misc.json", "w"), indent=2)
    print("done in %.1fs" % (time.time() - t0))
