# -*- coding: utf-8 -*-
"""Disposition-leakage audit.
The court's `reason` field is excluded; this audit checks whether the retained `facts` text ever
contains the court's CURRENT disposition (which would leak the label into the TF-IDF models).
`facts` legitimately recites the defendant's PRIOR convictions (which carry their own sentences),
so we distinguish a present-tense active sentencing of the defendant from prior-record recitations,
and we check for the disposition-section heading (the jumun), which marks the holding.

NOTE ON THE KOREAN TEXT IN THIS FILE. The corpus consists of Korean court judgments, so the
regular expressions below must match Korean surface forms verbatim and cannot be translated
without changing the audit result. Every comment in this file is in English only, and the
glossary above the patterns gives the English meaning of every token each pattern matches, in
the order the pattern lists them, so the audit can be verified without reading Korean.

Inputs : outputs/stage1_prepared.parquet, outputs/stage2_features_verified.parquet
Outputs: outputs/disposition_audit.csv  (+ regex list printed for the manuscript)
"""
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd
import config
OUT = config.OUTPUT_DIR

ids = set(pd.read_parquet(OUT / "stage2_features_verified.parquet")["id"])
prep = pd.read_parquet(OUT / "stage1_prepared.parquet").drop_duplicates("id")
facts = prep[prep["id"].isin(ids)][["id", "facts"]].reset_index(drop=True)
n = len(facts)
print(f"Audited {n} unique case `facts` texts.")

# GLOSSARY OF THE TOKENS MATCHED BY THE THREE PATTERNS BELOW,
# in the order each pattern lists them:
#   RE_ANY   imprisonment with labour, confinement without labour, fine, short-term detention,
#            pronouncement of sentence, "shall be punished with", suspended execution of
#            sentence
#   RE_DISPO "the defendant" with the object marker, then one of imprisonment with labour /
#            confinement without labour / fine / short-term detention, then "shall be punished
#            with" or "is hereby sentenced"
#   RE_JUMUN the jumun, lit. "main text": the heading of the operative holding of a Korean
#            judgment (the section that states the sentence)

# Any sentencing verb at all (expected high: prior-conviction recitations).
RE_ANY = re.compile(r"징역|금고|벌금|구류|선고|처한다|집행유예")
# CURRENT-disposition leakage: a present-tense active sentencing OF THE DEFENDANT, reading
# "the defendant ... imprisonment / confinement / a fine / detention ... shall be punished
# with" or "... is hereby sentenced", or else the holding-section heading.
# (The prior-record section narrates priors in the past tense, so it does not match.)
RE_DISPO = re.compile(r"피고인을[^.]{0,25}(?:징역|금고|벌금|구류)[^.]{0,30}(?:처한다|선고한다)")
RE_JUMUN = re.compile(r"주\s*문")   # the jumun (holding) heading, with optional inner space

any_hits = facts["facts"].fillna("").str.contains(RE_ANY).sum()
dispo_mask = facts["facts"].fillna("").apply(lambda t: bool(RE_DISPO.search(str(t))))
jumun_hits = facts["facts"].fillna("").str.contains(RE_JUMUN).sum()
n_dispo = int(dispo_mask.sum())

print(f"  facts containing ANY sentencing verb (incl. prior records): {int(any_hits)} "
      f"({100*any_hits/n:.1f}%)  -- expected, these are prior-conviction recitations")
print(f"  facts matching a CURRENT-disposition pattern (leakage candidates): {n_dispo} "
      f"({100*n_dispo/n:.2f}%)")
print(f"  facts containing the holding heading 'jumun': {int(jumun_hits)}")

# Inspect the leakage candidates to confirm they are prior-record context, not the current sentence.
if n_dispo:
    print("\n  Leakage-candidate excerpts (for manual confirmation):")
    for _, r in facts[dispo_mask].head(10).iterrows():
        m = RE_DISPO.search(str(r["facts"]))
        s = max(0, m.start() - 30)
        print(f"   id {r['id']}: ...{str(r['facts'])[s:m.end()+10]}...")

pd.DataFrame([{"audited": n, "any_sentencing_verb": int(any_hits),
               "current_disposition_candidates": n_dispo, "jumun_heading": int(jumun_hits)}]
             ).to_csv(OUT / "disposition_audit.csv", index=False, encoding="utf-8-sig")
print("\nRegex used (for the manuscript):")
print("  ANY    :", RE_ANY.pattern)
print("  DISPO  :", RE_DISPO.pattern)
print("  JUMUN  :", RE_JUMUN.pattern)
print("  (the Korean tokens in these patterns are glossed in the source of this script)")
print("\nDisposition audit done -> outputs/disposition_audit.csv")
