"""
features.py - Rule-based sentencing-factor extractor for Korean DUI facts.

Five ML-A features (research plan v4, Section 4.1; Hwang & Eom 2022 four fields + year):
  bac, distance_km, vehicle, prior_dui_count, offense_year.

LEAKAGE CONTROL (plan v4, Section 2): these extraction rules were designed by inspecting
the TRAIN split only and are FROZEN here; they are applied unchanged to valid/test/CV folds.
The rules are deterministic regular expressions, so application introduces no learned
parameters. Extraction quality is reported via a manual-validation sample in Stage 2.

NOTE ON THE KOREAN TEXT IN THIS FILE
------------------------------------
The corpus consists of Korean court judgments, so every regular expression below must match
Korean surface forms verbatim; those string literals are part of the method and cannot be
translated away without changing the results. Every comment in this file is in English only.
Where a pattern lists alternatives, the comment above it gives their English meanings in the
same order as the alternatives appear in the pattern, so a reader who does not read Korean can
follow and audit every rule from the comments alone.

Two structural markers of a Korean judgment recur below and are named here in English:
  the "criminal facts" heading   - opens the present-offense narrative; the text before it is
                                   the prior-conviction listing.
  the "the defendant ..." opener - the word for "the defendant" followed by the subject
                                   marker; opens a sentence narrating an act of the defendant.
"""
import re

# -- Blood alcohol concentration -------------------------------------------
# The pattern anchors on the word for "concentration" (the final element of the compound
# meaning "blood alcohol concentration"), then a decimal number, then either the "%" sign or
# the spelled-out word for "percent".
# It matches, for example, a blood-alcohol phrase followed by "0.095%", a bare number followed
# by the word for percent ("0.216"), and
# "0. 179%" (0.179% with a stray inner space, which the pattern tolerates).
# Past convictions rarely state a BAC, but when they do the present-offense BAC comes LAST,
# so take the last match.
_RE_BAC = re.compile(r"농도\s*([0-9]\s*\.\s*[0-9]+)\s*(?:%|퍼센트)")
def extract_bac(facts: str) -> float:
    ms = _RE_BAC.findall(facts)
    return float(ms[-1].replace(" ", "")) if ms else float("nan")

# -- Driving distance in km -------------------------------------------------
# The Korean literals in these two patterns are unit names only.
# Kilometre alternatives, in pattern order: the ASCII "km", the CJK single-character form of
# "km", then the three Korean spellings of "kilometre" found in the corpus.
# Metre alternatives, in pattern order: the ASCII "m", the Korean word for "metre", then the
# full-width Latin "m" used in Korean typesetting. Metres are divided by 1000.
# Typical distance phrases read "a stretch of about 300 m" or "a distance of about 10 m".
# The trailing metre unit is usually followed by a Korean grammatical particle meaning "of"
# rather than by whitespace, so no \b is used; a negative lookahead keeps the "m" of "km"
# from matching as a metre unit.
_RE_DIST_KM = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:km|㎞|킬로미터|키로미터|키로)", re.IGNORECASE)
_RE_DIST_M  = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:m|미터|ｍ)(?![a-zA-Z])")
def extract_distance_km(facts: str) -> float:
    m = _RE_DIST_KM.search(facts)
    if m:
        return float(m.group(1))
    m = _RE_DIST_M.search(facts)
    if m:
        return float(m.group(1)) / 1000.0
    return float("nan")

# -- Vehicle type (categorical) -------------------------------------------
# Ordered checks: motorcycle > truck > van/bus > passenger > other.
# The alternatives of each branch pattern, in the order they appear in that pattern:
#   motorcycle "two-wheeled", "motorbike", "motorised bicycle / moped", "scooter"
#   truck      "freight, cargo" (the first element of the compound for "cargo vehicle"),
#              then five light-truck model names common in Korea -- Porter, Bongo, Labo,
#              Damas, Mighty -- then "dump (truck)" and "truck"
#   van        "multi-passenger (van)", "bus"
#   passenger  "passenger (car)"
def extract_vehicle(facts: str) -> str:
    if re.search(r"이륜|오토바이|원동기|스쿠터", facts):
        return "motorcycle"
    # explicit "freight, cargo" or a common 1-ton truck model name
    if re.search(r"화물|포터|봉고|라보|다마스|마이티|덤프|트럭", facts):
        return "truck"
    if re.search(r"승합|버스", facts):
        return "van"
    if re.search(r"승용", facts):
        return "passenger"
    return "other"

# -- Prior alcohol-related driving record count ---------------------------
# DEFINITION (confirmed with domain expert): count every dated punishment entry in the
# criminal-record section that is alcohol-related driving -- BOTH drunk driving AND refusal
# of a breath test. All listed entries are counted (no offense-date filter). The record
# section is the text before the "criminal facts" heading when that tag is present (~35%);
# otherwise the text before the LAST "the defendant ..." sentence, which narrates the present
# offense. This sentence-based split needs no clock-time pattern and excludes the present
# offense date, so no adjustment is required. Counting dated events (not offense labels)
# handles a single label covering several dated convictions.
# The date pattern tolerates spaces around the dots ("2019. 6 . 25.") and an optional trailing
# dot ("2020. 8. 17" before a clock time).
_RE_RECORD_DATE = re.compile(r"(?:19|20)\d{2}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}\s*\.?")
# The three alcohol-related charge names, in tuple order: "drunk driving", "refusal of an
# alcohol test", and "refusal of testing", the short form used once the context is established
_ALCOHOL_TERMS = ("음주운전", "음주측정거부", "측정거부")
# Markers that occur in the PRESENT-offense sentence (never in the record listing), in
# pattern order:
#   "blood alcohol concentration", allowing for its spelling variants
#   "intoxicated, under the influence"
#   "alcohol testing"
#   "Widmark", i.e. the Widmark formula used to back-calculate BAC
_RE_PRESENT = re.compile(r"혈중알[코콜]+올?\s*농도|술에\s*취한|음주측정|위드마크")
def _record_section(facts: str) -> str:
    """Text containing only the prior-conviction listing (excludes the present offense)."""
    if "범죄사실" in facts:                       # the "criminal facts" heading
        return facts.split("범죄사실")[0]
    # The present offense is the LAST sentence carrying a BAC/test marker (a past conviction
    # can also state a BAC, so use the last marker, not the first). Cut at the
    # "the defendant ..." opener that begins that sentence.
    ms = list(_RE_PRESENT.finditer(facts))
    if ms:
        idx = facts.rfind("피고인은", 0, ms[-1].start())
    else:
        idx = facts.rfind("피고인은")
    if idx <= 0:
        # Only one "the defendant ..." opener in the whole text, as in "despite having a
        # prior record, the defendant drove ...". Split at the connector that closes the
        # record clause instead. The alternatives below, in pattern order: "despite being"
        # in its two spellings, "notwithstanding", "even so", and "has a prior record".
        m2 = re.search(r"있음에도|임에도|불구하고|그럼에도|전력이\s*있", facts)
        idx = m2.end() if m2 else 0
    return facts[:idx] if idx > 0 else ""
# A genuine prior DUI entry = a DATE whose nearby context carries BOTH a punishment word
# (which excludes parole / finality / execution-end / bare offense dates, none of which have
# one) AND an alcohol-driving charge. The charge may be named explicitly ("drunk driving",
# "refusal of testing") or referred to anaphorically by the fixed phrase meaning "the same
# offense" that _RE_SAME matches, which in a Korean record listing inherits the charge named
# in the preceding entry; _RE_SAME together with the `last_alcohol` state below implements
# that inheritance. An explicit non-alcohol charge ("obstruction of business", "fraud", ...)
# breaks the inheritance and drops the entry. A narrative count -- "N prior offenses of the
# same kind", "N same-kind prior offenses in total" -- overrides when larger.
# Punishment words, in pattern order: "fine", "imprisonment with labour", "confinement
# without labour", "summary order", "pronouncement of sentence", "short-term detention"
_RE_PUNISH_CTX = re.compile(r"벌금|징역|금고|약식|선고|구류")
_RE_ALCOHOL = re.compile(r"음주운전|음주측정거부|측정거부")
_RE_SAME = re.compile(r"같은\s*죄")            # "the same offense" (anaphoric charge reference)
# Non-alcohol charges, in pattern order: obstruction of business, fraud, theft, assault,
# bodily injury, indecent act by compulsion, sexual violence, narcotics, embezzlement,
# breach of trust, fleeing the scene, driving without a licence, destruction of property,
# obstruction of official duties
_RE_OTHER_CHARGE = re.compile(r"업무방해|사기|절도|폭행|상해|강제추행|성폭력|마약|"
                              r"횡령|배임|도주|무면허운전|재물손괴|공무집행방해")
# Narrative count: the phrase for "prior record of the same kind" (with an optional inner
# word for "offense"), or the word for "in total", then a number, then the occurrence counter
_RE_PRIOR_NARRATIVE = re.compile(r"(?:동종(?:\s*범죄)?\s*전력이?|총)\s*(\d+)\s*회")
# Procedural dates, not punishments, in pattern order: "(judgment) became final", "parole",
# "release", "completion", "elapse", "expiry"
_RE_PROC_DATE = re.compile(r"확정|가석방|석방|종료|경과|만료")
def extract_prior_dui_count(facts: str) -> int:
    record = _record_section(facts)
    dates = list(_RE_RECORD_DATE.finditer(record))
    # A charge label may precede the first listed date; seed the inheritance state from it.
    first = dates[0].start() if dates else len(record)
    last_alcohol = bool(_RE_ALCOHOL.search(record[:first]))
    n_dates = 0
    for m in dates:
        if _RE_PROC_DATE.search(record[m.end(): m.end() + 18]):
            continue  # finality / parole / execution-end date, not a conviction
        ctx = record[m.start(): m.end() + 45]
        if not _RE_PUNISH_CTX.search(ctx):
            continue  # not a punishment date
        if _RE_ALCOHOL.search(ctx):
            last_alcohol = True; n_dates += 1
        elif _RE_OTHER_CHARGE.search(ctx):
            last_alcohol = False  # explicit non-alcohol charge breaks inheritance
        elif _RE_SAME.search(ctx) or True:
            if last_alcohol:      # the anaphoric "same offense" case, or no charge stated,
                n_dates += 1      # so inherit the charge of the preceding entry
    mn = _RE_PRIOR_NARRATIVE.search(record)
    n_narrative = int(mn.group(1)) if mn else 0
    if n_dates == 0 and n_narrative == 0:
        return 0
    return max(n_dates, n_narrative)

# -- Offense year (last full date in facts = the offense datetime) ---------
# Trailing dot optional ("2020. 8. 17 02:25..."); spaces around the dots tolerated.
_RE_YEAR = re.compile(r"(20[0-9]{2})\s*\.\s*[0-9]{1,2}\s*\.\s*[0-9]{1,2}\s*\.?")
def extract_offense_year(facts: str):
    yrs = _RE_YEAR.findall(facts)
    return int(yrs[-1]) if yrs else None


def extract_features(facts: str) -> dict:
    """Return the five ML-A sentencing factors for one case's facts."""
    return {
        "bac": extract_bac(facts),
        "distance_km": extract_distance_km(facts),
        "vehicle": extract_vehicle(facts),
        "prior_dui_count": extract_prior_dui_count(facts),
        "offense_year": extract_offense_year(facts),
    }
