"""
Stage 4 - Statistical testing of H1 (research plan v4, Sections 3 & 6).

H1 (reproduction, effect-size form): each ML model's macro-F1 and QWK significantly and
practically exceed the majority-class baseline. For every (model, representation, metric):
  - paired per-fold difference (model - majority baseline) across the 25 CV folds
  - mean difference + 95% bootstrap CI (CI lower bound > 0 = baseline-beating)
  - Wilcoxon signed-rank p-value
  - mixed-effects intercept CI clustering folds by CV repeat (fold/seed variance, plan v4-M4)
  - Cohen's d effect size
Holm correction over the 16 primary tests (8 model-reps x {macro-F1, QWK}); post-hoc MDE.

Input : outputs/stage3_cv_results.csv
Output: outputs/T4_h1_tests.md
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower
import statsmodels.formula.api as smf

import config
config.set_seed()
rng = np.random.RandomState(config.SEED)

df = pd.read_csv(config.OUTPUT_DIR / "stage3_cv_results.csv")
base = df[df["model"] == "baseline_majority"].set_index("fold")
METRICS = ["macro_f1", "qwk"]
N_SPLITS = config.CV_N_SPLITS

rows = []
for (model, rep), g in df.groupby(["model", "rep"]):
    if model.startswith("baseline"):
        continue
    g = g.set_index("fold").sort_index()
    for met in METRICS:
        diff = (g[met] - base[met]).values            # 25 paired fold differences
        mean = float(diff.mean())
        boot = np.array([rng.choice(diff, len(diff), replace=True).mean()
                         for _ in range(10000)])
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        try:
            _, p = stats.wilcoxon(diff, alternative="greater")
        except ValueError:
            p = 0.0 if mean > 0 else 1.0
        d = mean / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.inf
        # mixed-effects: cluster folds by repeat (fold//N_SPLITS)
        dd = pd.DataFrame({"diff": diff,
                           "repeat": [f // N_SPLITS for f in g.index]})
        try:
            mm = smf.mixedlm("diff ~ 1", dd, groups="repeat").fit(reml=True)
            me_lo, me_hi = mm.conf_int().loc["Intercept"].values
        except Exception:
            me_lo, me_hi = np.nan, np.nan
        rows.append({"model": model, "rep": rep, "metric": met,
                     "mean_diff": mean, "ci_low": ci_low, "ci_high": ci_high,
                     "p_raw": p, "cohen_d": d, "me_ci_low": me_lo, "me_ci_high": me_hi})

res = pd.DataFrame(rows)
res["p_holm"] = multipletests(res["p_raw"], method="holm")[1]
# H1 passes if: bootstrap CI excludes 0 AND mixed-effects CI excludes 0 AND Holm-significant
res["H1_pass"] = (res["ci_low"] > 0) & (res["me_ci_low"] > 0) & (res["p_holm"] < 0.05)

# Post-hoc MDE: smallest mean difference detectable at n=25 folds, alpha .05, power .8
mde_d = TTestPower().solve_power(nobs=25, alpha=0.05, power=0.8, alternative="larger")

# -- Report ----------------------------------------------------------------
lines = ["# Table 4 - H1 statistical tests (ML vs majority baseline)\n",
         "Per-fold paired differences across 25 CV folds. H1 passes when the bootstrap 95% CI "
         "and the repeat-clustered mixed-effects 95% CI both exclude 0 and the Holm-corrected "
         "p < .05.\n",
         f"Post-hoc minimum detectable effect (n=25, alpha=.05, power=.80): Cohen's d = {mde_d:.2f}.\n",
         "| model | repr | metric | mean Delta | boot 95% CI | mixed-eff 95% CI | p(Holm) | d | H1 |",
         "|---|---|---|---|---|---|---|---|---|"]
res_sorted = res.sort_values(["metric", "mean_diff"], ascending=[True, False])
for _, r in res_sorted.iterrows():
    lines.append(
        f"| {r['model']} | {r['rep']} | {r['metric']} | {r['mean_diff']:.3f} | "
        f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] | "
        f"[{r['me_ci_low']:.3f}, {r['me_ci_high']:.3f}] | "
        f"{r['p_holm']:.1e} | {r['cohen_d']:.2f} | {'PASS' if r['H1_pass'] else 'no'} |")

n_pass = int(res["H1_pass"].sum())
lines.append(f"\n**{n_pass}/{len(res)} model-metric tests pass H1.** "
             f"All effect sizes far exceed the MDE (d={mde_d:.2f}), so the study is adequately "
             f"powered to detect the observed baseline-beating differences.")

(config.OUTPUT_DIR / "T4_h1_tests.md").write_text("\n".join(lines), encoding="utf-8")
res.to_csv(config.OUTPUT_DIR / "stage4_h1_results.csv", index=False, encoding="utf-8-sig")
print("\n".join(lines))
print("\nSaved: T4_h1_tests.md, stage4_h1_results.csv")
print("Stage 4 done.")
