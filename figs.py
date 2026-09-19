"""Figures for the PPCT post."""
import numpy as np, pandas as pd, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
sys.path.insert(0, ".")
import ppct

# ---- palette (validated: adjacent CVD dE 9.2, normal-vision dE 27.6, light mode)
C = {"Reference": "#2a78d6", "PPI": "#eb6834", "PPCT": "#1baf7a",
     "PPCT2": "#4a3aa7", "ANCOVA": "#e34948", "Naive (ML only)": "#898781"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURF = "#fcfcfb"
ORDER = ["Reference", "PPI", "PPCT", "PPCT2", "ANCOVA"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
    "figure.dpi": 150,
})


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ==========================================================================
def fig_design_curve(path="fig1_design_curve.png"):
    rho = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax[0].plot(rho, 1 - rho**2, color=C["PPCT"])
    ax[0].fill_between(rho, 1 - rho**2, 1, color=C["PPCT"], alpha=0.10)
    ax[0].set(title="What the prognostic score buys you",
              xlabel=r"correlation of prediction with outcome, $\rho$",
              ylabel="variance, relative to a standard RCT", ylim=(0, 1.05))
    for r in (0.5, 0.7, 0.9):
        ax[0].plot([r], [1 - r**2], "o", ms=7, color=C["PPCT"],
                   mec=SURF, mew=2, zorder=3)
        ax[0].annotate(f"ρ={r}: {100*(1-r**2):.0f}%", (r, 1 - r**2),
                       textcoords="offset points", xytext=(8, 10),
                       color=INK2, fontsize=10)

    ax[1].plot(rho, ppct.control_arm_ratio(rho), color=C["PPI"])
    ax[1].set(title="How small the control arm can get",
              xlabel=r"correlation in the control arm, $\rho_c$",
              ylabel=r"$n_*/m$  (controls per treated)", ylim=(0, 1.05))
    for r in (0.5, 0.7, 0.9):
        y = float(ppct.control_arm_ratio(r))
        ax[1].plot([r], [y], "o", ms=7, color=C["PPI"], mec=SURF, mew=2, zorder=3)
        ax[1].annotate(f"ρ={r}: {y:.2f}", (r, y), textcoords="offset points",
                       xytext=(8, 10), color=INK2, fontsize=10)
    for a in ax:
        a.xaxis.set_major_locator(MultipleLocator(0.2))
    finish(fig, path)


def fig_sampling(path="fig2_sampling.png"):
    sd = pd.read_csv("out_sampling.csv")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bins = np.linspace(0.6, 1.4, 55)
    for k in ["Reference", "PPI", "PPCT"]:
        ax.hist(sd[k], bins=bins, density=True, histtype="step",
                color=C[k], lw=2, label=f"{k}  (SD {sd[k].std():.3f})")
    ax.axvline(1.0, color=INK, lw=1.4, ls="--")
    ax.annotate("true ATE = 1", (1.0, ax.get_ylim()[1] * 0.95),
                xytext=(6, -2), textcoords="offset points", color=INK2, fontsize=10)
    ax.set(title="All three are unbiased. Only two are precise.",
           xlabel="estimated ATE", ylabel="density")
    ax.legend(loc="upper left", fontsize=10)
    fig.text(0.01, -0.04, "4,000 simulated trials · 100 treated, 100 controls · "
             "prognostic score correlated ρ = 0.9 with the outcome",
             color=MUTED, fontsize=9)
    finish(fig, path)


def fig_unbalanced(path="fig3_unbalanced.png"):
    ub = pd.read_csv("out_unbalanced.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for k in ORDER:
        d = ub[ub.estimator == k]
        ax[0].plot(d.n, d["sd"], "-o", ms=5, color=C[k], label=k, mec=SURF, mew=1.2)
    ax[0].set(xscale="log", yscale="log", xlabel="controls, n (treated m = 100)",
              ylabel="SD of the estimator",
              title="Shrinking the control arm")
    ax[0].set_xticks([2, 5, 10, 20, 50, 100])
    ax[0].set_xticklabels([2, 5, 10, 20, 50, 100])
    ax[0].legend(fontsize=9.5)

    ref = ub[ub.estimator == "Reference"].set_index("n")["sd"]
    for k in ["PPI", "PPCT", "PPCT2", "ANCOVA"]:
        d = ub[ub.estimator == k]
        ax[1].plot(d.n, d["sd"].values / ref[d.n].values, "-o", ms=5,
                   color=C[k], label=k, mec=SURF, mew=1.2)
    ax[1].axhline(1, color=INK, lw=1.2, ls="--")
    ax[1].set(xscale="log", yscale="log", xlabel="controls, n",
              ylabel="SD relative to Reference", title="Below 1 is a win")
    ax[1].set_yticks([0.4, 0.6, 1, 2, 4, 8, 12])
    ax[1].set_yticklabels(["0.4", "0.6", "1", "2", "4", "8", "12"])
    ax[1].set_xticks([2, 5, 10, 20, 50, 100])
    ax[1].set_xticklabels([2, 5, 10, 20, 50, 100])
    ax[1].legend(fontsize=9.5)
    fig.text(0.01, -0.04, "2,000 simulated trials per point · ρ = 0.9 · "
             "Gaussian outcome", color=MUTED, fontsize=9)
    finish(fig, path)


def fig_misspec(path="fig4_misspec.png"):
    df = pd.read_csv("out_misspec2.csv")
    names = ["Naive (ML only)"] + ORDER
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, k in enumerate(names):
        v = df[k].values
        ax.scatter(v, np.full_like(v, i) + np.random.uniform(-.16, .16, len(v)),
                   s=3, alpha=.18, color=C[k], edgecolors="none")
        ax.plot([v.mean()], [i], "|", ms=26, mew=3, color=INK)
        ax.annotate(f"mean {v.mean():.2f} · SD {v.std(ddof=1):.2f}",
                    (v.mean(), i), xytext=(0, 14), textcoords="offset points",
                    ha="center", color=INK2, fontsize=9.5)
    ax.axvline(2.0, color=INK, lw=1.4, ls="--")
    ax.annotate("true ATE = 2", (2.0, len(names) - 0.45), xytext=(7, 0),
                textcoords="offset points", color=INK2, fontsize=10)
    ax.set_yticks(range(len(names)), names)
    ax.set(xlabel="estimated ATE", title="A badly wrong model, and an estimator that survives it",
           ylim=(-0.8, len(names) - 0.3))
    ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.05, "2,000 simulated trials · prognostic model is linear while the truth "
             "is not, and mis-calibrated by +15 units", color=MUTED, fontsize=9)
    finish(fig, path)


def fig_power(path="fig5_power.png"):
    pw = pd.read_csv("out_power.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for k in ORDER:
        d = pw[pw.estimator == k]
        ax.plot(d.rho, d.power, "-o", ms=4.5, color=C[k], label=k, mec=SURF, mew=1)
    ax.set(xlabel=r"quality of the prognostic score, $\rho$", ylabel="power at α = 0.05",
           title="Power, as a function of how good your model is", ylim=(0, 1.02))
    ax.legend(fontsize=9.5, loc="upper left")
    fig.text(0.01, -0.04, "2,000 simulated trials per point · 100 treated, 100 controls · "
             "true ATE = 0.3 SD", color=MUTED, fontsize=9)
    finish(fig, path)


def _forest(ax, res, title, note=None, unit=""):
    ys = np.arange(len(res))[::-1]
    for y, (_, r) in zip(ys, res.iterrows()):
        c = C[r.estimator]
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=c, lw=2.4,
                solid_capstyle="round")
        ax.plot([r.ate], [y], "o", ms=9, color=c, mec=SURF, mew=2, zorder=3)
        ax.annotate(f"{r.ate:.2f}  ({r.ci_lo:.2f}, {r.ci_hi:.2f})",
                    (1.02, y), xycoords=("axes fraction", "data"),
                    xytext=(0, -4), textcoords="offset points",
                    color=INK2, fontsize=9.5)
    ax.axvline(0, color=INK, lw=1.4, ls="--")
    ax.set_yticks(ys, res.estimator)
    ax.set(title=title, xlabel=f"estimated ATE{unit}")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.06)
    if note:
        ax.annotate(note, (0.0, -0.20), xycoords="axes fraction",
                    color=MUTED, fontsize=9, va="top")


def fig_actg(path="fig6_actg.png"):
    res = pd.read_csv("out_actg_estimators.csv")
    res = res.set_index("estimator").loc[ORDER].reset_index()
    meta = json.load(open("out_real.json"))["actg_meta"]
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    _forest(ax, res, "ACTG 175: effect on CD4 count at 20 weeks",
            note=f"{meta['m']} treated · {meta['n']} controls · "
                 f"out-of-sample R² of the prognostic score = {meta['r2_oos_control']:.2f} "
                 f"(ρ_c = {meta['rho_c']:.2f})", unit="  (CD4 cells/mm³)")
    finish(fig, path)


def fig_actg_shrink(path="fig7_actg_shrink.png"):
    ub = pd.read_csv("out_actg_unbalanced.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for k in ["Reference", "PPCT"]:
        d = ub[ub.estimator == k]
        ax.plot(d.n_control, d.sd_empirical, "-o", ms=6, color=C[k],
                label=k, mec=SURF, mew=1.4)
    ref = ub[ub.estimator == "Reference"]
    ppc = ub[ub.estimator == "PPCT"]
    # equivalence annotation: which n does PPCT need to match Reference at n=200?
    target = float(ref[ref.n_control == 200].sd_empirical.iloc[0])
    ax.axhline(target, color=MUTED, lw=1, ls=":")
    ax.annotate(f"Reference with 200 controls\n= PPCT with ≈{int(np.interp(-target, -ppc.sd_empirical, ppc.n_control))} controls",
                (12, target), xytext=(0, 10), textcoords="offset points",
                color=INK2, fontsize=9.5)
    ax.set(xscale="log", xlabel="controls kept in the trial, n (treated m = 200)",
           ylabel="SD of the estimated ATE (CD4 cells/mm³)",
           title="ACTG 175: how much control arm do you actually need?")
    ax.set_xticks([10, 20, 40, 80, 120, 200])
    ax.set_xticklabels([10, 20, 40, 80, 120, 200])
    ax.legend(fontsize=10)
    fig.text(0.01, -0.04, "600 resampled sub-trials per point, drawn from the real ACTG 175 data",
             color=MUTED, fontsize=9)
    finish(fig, path)


def fig_nhanes(path="fig8_nhanes.png"):
    res = pd.read_csv("out_nhanes_estimators.csv")
    res = res.set_index("estimator").loc[ORDER].reset_index()
    r = json.load(open("out_real.json"))
    meta = r["nhanes_meta"]
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    _forest(ax, res, "NHANES: physically active vs not, systolic blood pressure",
            note=f"{meta['m']} active · {meta['n']} inactive · NOT randomised — "
                 f"the active group is {r['nhanes_age_by_arm']['0'] - r['nhanes_age_by_arm']['1']:.1f} years younger "
                 f"and {r['nhanes_bmi_by_arm']['0'] - r['nhanes_bmi_by_arm']['1']:.1f} BMI units lighter",
            unit="  (mmHg)")
    finish(fig, path)


def fig_actg_scatter(path="fig9_actg_scatter.png"):
    from real_data import actg_setup
    _, _, trial, _, _ = actg_setup()
    c = trial[trial.treat == 0]
    t = trial[trial.treat == 1]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.scatter(c.f, c.cd420, s=14, alpha=.5, color=C["Reference"],
               edgecolors="none", label="control (ZDV only)")
    ax.scatter(t.f, t.cd420, s=14, alpha=.35, color=C["PPI"],
               edgecolors="none", label="treated")
    lim = [min(trial.f.min(), 50), max(trial.f.max(), 900)]
    ax.plot(lim, lim, color=INK, lw=1.2, ls="--")
    ax.annotate("perfect prediction", (lim[1], lim[1]), xytext=(-8, 8),
                textcoords="offset points", ha="right", color=INK2, fontsize=9.5)
    ax.set(xlabel="prognostic score f(X) from the historical model",
           ylabel="observed CD4 at 20 weeks",
           title="The prognostic score is good, not great")
    ax.legend(fontsize=10, loc="upper left")
    finish(fig, path)


if __name__ == "__main__":
    fig_design_curve(); fig_sampling(); fig_unbalanced(); fig_misspec()
    fig_power(); fig_actg(); fig_actg_shrink(); fig_nhanes(); fig_actg_scatter()
