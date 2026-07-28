"""
features.py - Rule-based sentencing-factor extractor for Korean DUI facts.

Five ML-A features (research plan v4, Section 4.1; Hwang & Eom 2022 four fields + year):
  bac, distance_km, vehicle, prior_dui_count, offense_year.

LEAKAGE CONTROL (plan v4, Section 2): these extraction rules were designed by inspecting
the TRAIN split only and are FROZEN here; they are applied unchanged to valid/test/CV folds.
The rules are deterministic regular expressions, so application introduces no learned
parameters. Extraction quality is reported via a manual-validation sample in Stage 2.
"""
import re

# -- Blood alcohol concentration (e.g. "혈중알콜농도 0.095%", "0.216퍼센트", "0. 179%") --
# Accept "%" or "퍼센트", and tolerate an inner space ("0. 179"). Past convictions rarely
# state a BAC, but when they do the present-offense BAC comes LAST, so take the last match.
_RE_BAC = re.compile(r"농도\s*([0-9]\s*\.\s*[0-9]+)\s*(?:%|퍼센트)")
def extract_bac(facts: str) -> float:
    ms = _RE_BAC.findall(facts)
    return float(ms[-1].replace(" ", "")) if ms else float("nan")

# -- Driving distance in km -------------------------------------------------
# Normalize km / m variants: "km", "Km", "㎞", "킬로미터", "키로미터", "키로" (km);
# "m", "미터", "ｍ" as meters (e.g. "약 300m의 구간", "약 10m의 거리") -> /1000.
# The trailing meter unit is often followed by a Korean particle ("m의"), so no \b is used;
# a negative lookahead keeps the "m" of "km" from matching here.
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
def extract_vehicle(facts: str) -> str:
    if re.search(r"이륜|오토바이|원동기|스쿠터", facts):
        return "motorcycle"
    # explicit "화물" or common 1-ton truck model names
    if re.search(r"화물|포터|봉고|라보|다마스|마이티|덤프|트럭", facts):
        return "truck"
    if re.search(r"승합|버스", facts):
        return "van"
    if re.search(r"승용", facts):
        return "passenger"
    return "other"

# -- Prior alcohol-related driving record count ---------------------------
# DEFINITION (confirmed with domain expert): count every dated punishment entry in the
# criminal-record section that is alcohol-related driving -- BOTH drunk driving (음주운전)
# AND refusal of a breath test (음주측정거부). All listed entries are counted (no offense-
# date filter). The record section is the text before "범죄사실" when that tag is present
# (~35%); otherwise the text before the LAST "피고인은" sentence (which narrates the present
# offense). This sentence-based split needs no clock-time pattern and excludes the present
# offense date, so no adjustment is required. Counting dated events (not offense labels)
# handles a single label covering several dated convictions.
# Date pattern tolerates spaces around dots ("2019. 6 . 25.") and an optional trailing dot
# ("2020. 8. 17" before a clock time).
_RE_RECORD_DATE = re.compile(r"(?:19|20)\d{2}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}\s*\.?")
_ALCOHOL_TERMS = ("음주운전", "음주측정거부", "측정거부")
# Markers that occur in the PRESENT-offense sentence (never in the record listing).
_RE_PRESENT = re.compile(r"혈중알[코콜]+올?\s*농도|술에\s*취한|음주측정|위드마크")
def _record_section(facts: str) -> str:
    """Text containing only the prior-conviction listing (excludes the present offense)."""
    if "범죄사실" in facts:
        return facts.split("범죄사실")[0]
    # The present offense is the LAST sentence with a BAC/test marker (a past conviction can
    # also state a BAC, so use the last marker, not the first). Cut at the "피고인은" before it.
    ms = list(_RE_PRESENT.finditer(facts))
    if ms:
        idx = facts.rfind("피고인은", 0, ms[-1].start())
    else:
        idx = facts.rfind("피고인은")
    if idx <= 0:
        # single-"피고인은" sentence ("...전력이 있음에도, 피고인은 ... 운전"): split at the
        # record-ending connector instead.
        m2 = re.search(r"있음에도|임에도|불구하고|그럼에도|전력이\s*있", facts)
        idx = m2.end() if m2 else 0
    return facts[:idx] if idx > 0 else ""
# A genuine prior DUI entry = a DATE whose nearby context has BOTH a punishment word
# (avoids parole/confirmation/execution-end/offense-date entries, which lack one) AND an
# alcohol-driving charge (음주운전/측정거부, or "같은 죄" inheriting the prior alcohol charge;
# this drops unrelated priors like 업무방해/사기). Narrative count ("동종전력이 N회",
# "총 N회의 동종전력") overrides when larger.
_RE_PUNISH_CTX = re.compile(r"벌금|징역|금고|약식|선고|구류")
_RE_ALCOHOL = re.compile(r"음주운전|음주측정거부|측정거부")
_RE_SAME = re.compile(r"같은\s*죄")
_RE_OTHER_CHARGE = re.compile(r"업무방해|사기|절도|폭행|상해|강제추행|성폭력|마약|"
                              r"횡령|배임|도주|무면허운전|재물손괴|공무집행방해")
_RE_PRIOR_NARRATIVE = re.compile(r"(?:동종(?:\s*범죄)?\s*전력이?|총)\s*(\d+)\s*회")
_RE_PROC_DATE = re.compile(r"확정|가석방|석방|종료|경과|만료")   # procedural, not a punishment
def extract_prior_dui_count(facts: str) -> int:
    record = _record_section(facts)
    dates = list(_RE_RECORD_DATE.finditer(record))
    # A charge label may precede the first listed date; seed the inheritance state from it.
    first = dates[0].start() if dates else len(record)
    last_alcohol = bool(_RE_ALCOHOL.search(record[:first]))
    n_dates = 0
    for m in dates:
        if _RE_PROC_DATE.search(record[m.end(): m.end() + 18]):
            continue  # confirmation / parole / execution-end date, not a conviction
        ctx = record[m.start(): m.end() + 45]
        if not _RE_PUNISH_CTX.search(ctx):
            continue  # not a punishment date
        if _RE_ALCOHOL.search(ctx):
            last_alcohol = True; n_dates += 1
        elif _RE_OTHER_CHARGE.search(ctx):
            last_alcohol = False  # explicit non-alcohol charge breaks inheritance
        elif _RE_SAME.search(ctx) or True:
            if last_alcohol:      # "같은 죄" or no charge stated -> inherit prior charge
                n_dates += 1
    mn = _RE_PRIOR_NARRATIVE.search(record)
    n_narrative = int(mn.group(1)) if mn else 0
    if n_dates == 0 and n_narrative == 0:
        return 0
    return max(n_dates, n_narrative)

# -- Offense year (last full date in facts = the offense datetime) ---------
# Trailing dot optional ("2020. 8. 17 02:25..."); spaces around dots tolerated.
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
