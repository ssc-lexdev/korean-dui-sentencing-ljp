"""
Stage 7 - H2 test: ML vs LLM (research plan v4, Section 4.3).

H2 = "ML outperforms zero-shot LLM". We pair the best ML model (Stage 3 OOF predictions)
against each Stage 6 LLM config on the SAME 1,500 unique cases and run McNemar's paired test.

Two correctness definitions are tested, because the Stage 6 distribution check showed that
raw accuracy is misleading for these ordinal labels (a model can collapse onto the majority
class and look "accurate", or collapse onto a minority class and look terrible):
  (1) EXACT   - prediction == true label              (primary, standard McNemar)
  (2) ORDINAL - |prediction - true| <= 1 (within one ordinal step)   (secondary, robustness)

For each correctness definition we Holm-correct across the family of ML-vs-LLM comparisons.
We also report each config's accuracy, macro-F1 and QWK for context (QWK/macro-F1 are the
fair metrics for ML-vs-LLM; raw accuracy is reported only to expose the artifact).

Inputs : outputs/stage3_oof_predictions.parquet, outputs/stage6_llm_predictions.parquet
Outputs: outputs/T7_h2_tests.md, outputs/stage7_h2_results.csv
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

import config
config.set_seed()
OUT = config.OUTPUT_DIR
LABELS = list(range(config.N_CLASSES))

# -- Load -----------------------------------------------------------------
oof = pd.read_parquet(OUT / "stage3_oof_predictions.parquet")   # id, model, rep, y_true, y_pred
llm = pd.read_parquet(OUT / "stage6_llm_predictions.parquet")   # id, model, variant, y_true, y_pred

# Best ML = SVM + TF-IDF (Stage 3: best macro-F1 0.451 / QWK 0.348).
BEST_ML = ("svm", "tfidf")
ml = (oof[(oof.model == BEST_ML[0]) & (oof.rep == BEST_ML[1])]
      [["id", "y_true", "y_pred"]]
      .rename(columns={"y_pred": "ml_pred"})
      .drop_duplicates("id"))
ml_name = f"{BEST_ML[0]}+{BEST_ML[1]}"

LLM_CONFIGS = [   # (model, variant, display)  - zero-shot are the primary H2 targets
    ("gpt",  "zero_shot_closed_book", "GPT zero-shot"),
    ("qwen", "zero_shot_closed_book", "Qwen zero-shot"),
    ("gpt",  "few_shot_open_book",    "GPT few-shot"),
    ("qwen", "few_shot_open_book",    "Qwen few-shot"),
]


def metrics(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    return (round((yp == yt).mean() * 100, 1),
            round(f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0), 3),
            round(cohen_kappa_score(yt, yp, labels=LABELS, weights="quadratic"), 3))


def correct(yp, yt, mode):
    yp, yt = np.asarray(yp), np.asarray(yt)
    return (yp == yt) if mode == "exact" else (np.abs(yp - yt) <= 1)


def run_mcnemar(ml_ok, llm_ok):
    # contingency: rows=ML correct?, cols=LLM correct?
    n11 = int(np.sum(ml_ok & llm_ok))     # both correct
    n10 = int(np.sum(ml_ok & ~llm_ok))    # ML right, LLM wrong  (b - favours ML)
    n01 = int(np.sum(~ml_ok & llm_ok))    # ML wrong, LLM right  (c - favours LLM)
    n00 = int(np.sum(~ml_ok & ~llm_ok))   # both wrong
    table = [[n11, n10], [n01, n00]]
    disc = n10 + n01
    # exact binomial when discordant count is small, else chi-square w/ continuity correction
    res = mcnemar(table, exact=(disc < 25), correction=True)
    return n10, n01, float(res.statistic), float(res.pvalue)


rows = []
for mode in ["exact", "ordinal"]:
    fam = []
    for lm_model, lm_variant, disp in LLM_CONFIGS:
        g = llm[(llm.model == lm_model) & (llm.variant == lm_variant)][["id", "y_true", "y_pred"]]
        g = g.rename(columns={"y_pred": "llm_pred"})
        m = ml.merge(g, on="id", how="inner")
        assert (m.y_true_x == m.y_true_y).all(), "y_true mismatch between ML and LLM tables"
        yt = m.y_true_x.to_numpy()
        ml_ok = correct(m.ml_pred.to_numpy(), yt, mode)
        llm_ok = correct(m.llm_pred.to_numpy(), yt, mode)
        b, c, stat, p = run_mcnemar(ml_ok, llm_ok)
        acc_ml = round(ml_ok.mean() * 100, 1)
        acc_llm = round(llm_ok.mean() * 100, 1)
        rows.append(dict(mode=mode, ml=ml_name, llm=disp, n=len(m),
                         ml_correct_pct=acc_ml, llm_correct_pct=acc_llm,
                         b_ml_right_llm_wrong=b, c_ml_wrong_llm_right=c,
                         mcnemar_stat=round(stat, 3), p_raw=p,
                         favours=("ML" if b > c else "LLM" if c > b else "tie")))
        fam.append((len(rows) - 1, p))
    # Holm within this correctness family
    idx, praw = zip(*fam)
    rej, padj, *_ = multipletests(praw, alpha=0.05, method="holm")
    for k, i in enumerate(idx):
        rows[i]["p_holm"] = float(padj[k])
        rows[i]["sig_holm_0.05"] = bool(rej[k])

res = pd.DataFrame(rows)
res.to_csv(OUT / "stage7_h2_results.csv", index=False)

# Context metrics per config (fair metrics: macro-F1 / QWK)
ctx = []
ml_full = ml.rename(columns={"ml_pred": "y_pred"})
a, f, q = metrics(ml_full.y_true, ml_full.y_pred)
ctx.append((ml_name + " (best ML)", a, f, q))
for lm_model, lm_variant, disp in LLM_CONFIGS:
    g = llm[(llm.model == lm_model) & (llm.variant == lm_variant)]
    a, f, q = metrics(g.y_true, g.y_pred)
    ctx.append((disp, a, f, q))
ctx = pd.DataFrame(ctx, columns=["model", "raw_acc", "macroF1", "QWK"])

# -- Report ----------------------------------------------------------------
def fmt_p(p):
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"

lines = []
lines.append(f"# Table 7 - H2 test: ML vs LLM (McNemar, best ML = {ml_name})\n")
lines.append("Paired McNemar on the same 1,500 unique cases. b = ML right / LLM wrong "
             "(favours ML); c = ML wrong / LLM right (favours LLM). Holm-corrected within "
             "each correctness family. EXACT = prediction==true; ORDINAL = within one step "
             "(|pred-true|<=1), included because raw accuracy is misleading for these labels "
             "(Stage 6 distribution check).\n")

lines.append("## Context metrics (fair comparison uses macro-F1 / QWK, not raw accuracy)\n")
lines.append("| model | raw_acc % | macro-F1 | QWK |")
lines.append("|---|---|---|---|")
for _, r in ctx.iterrows():
    lines.append(f"| {r['model']} | {r['raw_acc']} | {r['macroF1']} | {r['QWK']} |")
lines.append("")

for mode in ["exact", "ordinal"]:
    sub = res[res["mode"] == mode]
    lines.append(f"## McNemar - {mode.upper()} correctness\n")
    lines.append("| ML | LLM | ML corr % | LLM corr % | b | c | stat | p (Holm) | sig | favours |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in sub.iterrows():
        lines.append(f"| {r['ml']} | {r['llm']} | {r['ml_correct_pct']} | {r['llm_correct_pct']} "
                     f"| {r['b_ml_right_llm_wrong']} | {r['c_ml_wrong_llm_right']} "
                     f"| {r['mcnemar_stat']} | {fmt_p(r['p_holm'])} "
                     f"| {'YES' if r['sig_holm_0.05'] else 'no'} | {r['favours']} |")
    lines.append("")

# H2 verdict (primary = zero-shot, exact correctness)
zs = res[(res["mode"] == "exact") & (res["llm"].str.contains("zero-shot"))]
ml_wins_zs = ((zs["favours"] == "ML") & zs["sig_holm_0.05"]).sum()
lines.append("## H2 verdict\n")
lines.append(f"- Primary (EXACT correctness vs **zero-shot** LLMs): ML significantly beats "
             f"{ml_wins_zs}/{len(zs)} zero-shot configs (Holm < 0.05).")
best_llm_row = res[(res["mode"] == "exact") & (res["llm"] == "GPT few-shot")].iloc[0]
lines.append(f"- Hardest case (best LLM = GPT few-shot, EXACT): favours {best_llm_row['favours']}, "
             f"Holm p = {fmt_p(best_llm_row['p_holm'])} "
             f"({'sig' if best_llm_row['sig_holm_0.05'] else 'n.s.'}).")
lines.append("- NOTE: raw-accuracy ties (e.g. Qwen zero-shot) are majority-collapse artifacts; "
             "see macro-F1 / QWK in the context table and docs/results/stage6_distribution_check.md.")
lines.append("")
lines.append("Outputs: stage7_h2_results.csv (this table, machine-readable).")

(OUT / "T7_h2_tests.md").write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print("\nStage 7 done. -> outputs/T7_h2_tests.md, outputs/stage7_h2_results.csv")
