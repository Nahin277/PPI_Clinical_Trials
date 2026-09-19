"""How wrong can the prognostic model be?  A misspecification stress test."""
import numpy as np, pandas as pd, json, sys
sys.path.insert(0, ".")
import ppct
from sklearn.linear_model import LinearRegression


def world(N, rng, delta=2.0, p=5, noise=2.0):
    """Truth is non-linear in X; treatment shifts it by delta."""
    X = rng.normal(0, 1, (N, p))
    mu = 3 * np.sin(1.5 * X[:, 0]) + 1.2 * X[:, 1] ** 2 - 1.5 * X[:, 2] * X[:, 3] + X[:, 4]
    T = rng.binomial(1, 0.5, N)
    Y = mu + delta * T + rng.normal(0, noise, N)
    return X, Y, T, mu


def experiment(reps=2000, n_hist=600, N=400, delta=2.0, shift=15.0, seed=11):
    """Historical cohort -> linear model -> apply to a fresh trial.

    Two things are deliberately wrong with the model:
      1. it is LINEAR while the truth is not (structural misspecification);
      2. the historical cohort is on a different scale, so predictions are
         shifted by `shift` units (calibration failure).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(reps):
        Xh, Yh, Th, _ = world(n_hist, rng, delta)
        h = Th == 0                                   # historical CONTROLS only
        lm = LinearRegression().fit(Xh[h], Yh[h] + shift)

        X, Y, T, mu = world(N, rng, delta)
        f = lm.predict(X)

        r = ppct.run_all(Y, T, f)
        naive = Y[T == 1].mean() - f.mean()           # "digital twin" only
        rows.append({**{k: v.estimate for k, v in r.items()},
                     "Naive (ML only)": naive,
                     "lambda": r["PPCT"].lam,
                     "rho": np.corrcoef(f, Y)[0, 1],
                     "pred_bias": (f - mu).mean()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = experiment()
    df.to_csv("out_misspec2.csv", index=False)
    summary = df.agg(["mean", "std"]).T.round(3)
    print("True ATE = 2.0\n")
    print(summary.to_string())
    json.dump({"true_ate": 2.0,
               "summary": summary.to_dict(),
               }, open("out_misspec2.json", "w"), indent=2, default=float)
