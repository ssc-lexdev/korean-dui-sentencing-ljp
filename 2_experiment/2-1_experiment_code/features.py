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
Korean surface forms verbatim; the Korean string literals are part of the method and cannot be
translated away without changing the results. All comments are in English, and every Korean
token used in a pattern is glossed in English immediately above the pattern that uses it, so a
reader who does not read Korean can follow and audit each rule from the glosses alone.

Two section terms recur throughout:
  범죄사실 (beomjoe-sasil) - "criminal facts"; the heading that opens the present-offense
                            narrative in a Korean judgment. Text before it is the prior record.
  피고인은  (pigoin-eun)    - "the defendant [subject marker]"; opens a sentence narrating an
                            act of the defendant.
"""
import re

# -- Blood alcohol concentration -------------------------------------------
# Korean tokens in the pattern:
#   농도   "concentration" (as in 혈중알코올농도, "blood alcohol concentration")
#   퍼센트 "percent", the spelled-out alternative to the "%" sign
# Matches e.g. "혈중알콜농도 0.095%" (BAC 0.095%), "0.216퍼센트" (0.216 percent), and
# "0. 179%" (0.179% with a stray inner space, which the pattern tolerates).
# Past convictions rarely state a BAC, but when they do the present-offense BAC comes LAST,
# so take the last match.
_RE_BAC = re.compile(r"농도\s*([0-9]\s*\.\s*[0-9]+)\s*(?:%|퍼센트)")
def extract_bac(facts: str) -> float:
    ms = _RE_BAC.findall(facts)
    return float(ms[-1].replace(" ", "")) if ms else float("nan")

# -- Driving distance in km -------------------------------------------------
# Korean tokens in the patterns are unit names only:
#   킬로미터 / 키로미터 / 키로  "kilometre", in the three spellings found in the corpus;
#                              ㎞ is the CJK single-character form of "km"
#   미터  "metre";  ｍ is the full-width Latin "m" used in Korean typesetting
# Metres are divided by 1000. Typical distance phrases: "약 300m의 구간" ("a stretch of about
# 300 m"), "약 10m의 거리" ("a distance of about 10 m").
# The trailing metre unit is usually followed by a Korean particle -- 의 ("of") in "m의" --
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
# Korean tokens, by branch:
#   motorcycle 이륜 "two-wheeled", 오토바이 "motorbike", 원동기 "motorised bicycle / moped",
#              스쿠터 "scooter"
#   truck      화물 "freight, cargo" (as in 화물차, "cargo vehicle"), 트럭 "truck",
#              덤프 "dump (truck)", plus five light-truck model names common in Korea:
#              포터 Porter, 봉고 Bongo, 라보 Labo, 다마스 Damas, 마이티 Mighty
#   van        승합 "multi-passenger (van)", 버스 "bus"
#   passenger  승용 "passenger (car)"
def extract_vehicle(facts: str) -> str:
    if re.search(r"이륜|오토바이|원동기|스쿠터", facts):
        return "motorcycle"
    # explicit 화물 ("freight, cargo") or a common 1-ton truck model name
    if re.search(r"화물|포터|봉고|라보|다마스|마이티|덤프|트럭", facts):
        return "truck"
    if re.search(r"승합|버스", facts):
        return "van"
    if re.search(r"승용", facts):
        return "passenger"
    return "other"

# -- Prior alcohol-related driving record count ---------------------------
# DEFINITION (confirmed with domain expert): count every dated punishment entry in the
# criminal-record section that is alcohol-related driving -- BOTH drunk driving
# (음주운전, "drunk driving") AND refusal of a breath test (음주측정거부, "refusal of an
# alcohol test"). All listed entries are counted (no offense-date filter). The record section
# is the text before 범죄사실 ("criminal facts") when that tag is present (~35%); otherwise the
# text before the LAST 피고인은 ("the defendant ...") sentence, which narrates the present
# offense. This sentence-based split needs no clock-time pattern and excludes the present
# offense date, so no adjustment is required. Counting dated events (not offense labels)
# handles a single label covering several dated convictions.
# The date pattern tolerates spaces around the dots ("2019. 6 . 25.") and an optional trailing
# dot ("2020. 8. 17" before a clock time).
_RE_RECORD_DATE = re.compile(r"(?:19|20)\d{2}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}\s*\.?")
# 음주운전 "drunk driving" | 음주측정거부 "refusal of an alcohol test" | 측정거부 "refusal of
# testing", the short form used once the alcohol context is established
_ALCOHOL_TERMS = ("음주운전", "음주측정거부", "측정거부")
# Markers that occur in the PRESENT-offense sentence (never in the record listing):
#   혈중알코올농도 (spelling variants) "blood alcohol concentration"
#   술에 취한 "intoxicated, under the influence"
#   음주측정 "alcohol testing"
#   위드마크 "Widmark", i.e. the Widmark formula used to back-calculate BAC
_RE_PRESENT = re.compile(r"혈중알[코콜]+올?\s*농도|술에\s*취한|음주측정|위드마크")
def _record_section(facts: str) -> str:
    """Text containing only the prior-conviction listing (excludes the present offense)."""
    if "범죄사실" in facts:                       # the "criminal facts" heading
        return facts.split("범죄사실")[0]
    # The present offense is the LAST sentence carrying a BAC/test marker (a past conviction
    # can also state a BAC, so use the last marker, not the first). Cut at the 피고인은
    # ("the defendant ...") that opens that sentence.
    ms = list(_RE_PRESENT.finditer(facts))
    if ms:
        idx = facts.rfind("피고인은", 0, ms[-1].start())
    else:
        idx = facts.rfind("피고인은")
    if idx <= 0:
        # Only one 피고인은 in the whole text, as in "...전력이 있음에도, 피고인은 ... 운전"
        # ("despite having a prior record, the defendant drove ..."). Split at the connector
        # that closes the record clause instead:
        #   있음에도 / 임에도 / 그럼에도 "despite (being)", 불구하고 "notwithstanding",
        #   전력이 있 "has a prior record"
        m2 = re.search(r"있음에도|임에도|불구하고|그럼에도|전력이\s*있", facts)
        idx = m2.end() if m2 else 0
    return facts[:idx] if idx > 0 else ""
# A genuine prior DUI entry = a DATE whose nearby context carries BOTH a punishment word
# (which excludes parole / finality / execution-end / bare offense dates, none of which have
# one) AND an alcohol-driving charge. The charge may be named explicitly (음주운전 "drunk
# driving", 측정거부 "refusal of testing") or referred to anaphorically as 같은 죄 ("the same
# offense"), which in a Korean record listing inherits the charge named in the preceding entry;
# _RE_SAME together with the `last_alcohol` state below implements that inheritance. An
# explicit non-alcohol charge (업무방해 "obstruction of business", 사기 "fraud", ...) breaks the
# inheritance and drops the entry. A narrative count -- 동종전력이 N회 ("N prior offenses of the
# same kind"), 총 N회의 동종전력 ("N same-kind prior offenses in total") -- overrides when larger.
# Punishment words: 벌금 "fine", 징역 "imprisonment with labour", 금고 "confinement without
#                   labour", 약식 "summary order", 선고 "pronouncement of sentence",
#                   구류 "short-term detention"
_RE_PUNISH_CTX = re.compile(r"벌금|징역|금고|약식|선고|구류")
_RE_ALCOHOL = re.compile(r"음주운전|음주측정거부|측정거부")
_RE_SAME = re.compile(r"같은\s*죄")            # "the same offense" (anaphoric charge reference)
# Non-alcohol charges, in pattern order: 업무방해 obstruction of business, 사기 fraud,
# 절도 theft, 폭행 assault, 상해 bodily injury, 강제추행 indecent act by compulsion,
# 성폭력 sexual violence, 마약 narcotics, 횡령 embezzlement, 배임 breach of trust,
# 도주 fleeing the scene, 무면허운전 driving without a licence, 재물손괴 destruction of
# property, 공무집행방해 obstruction of official duties
_RE_OTHER_CHARGE = re.compile(r"업무방해|사기|절도|폭행|상해|강제추행|성폭력|마약|"
                              r"횡령|배임|도주|무면허운전|재물손괴|공무집행방해")
# 동종(범죄)전력 "prior record of the same kind" | 총 "in total" | 회 counter for occurrences
_RE_PRIOR_NARRATIVE = re.compile(r"(?:동종(?:\s*범죄)?\s*전력이?|총)\s*(\d+)\s*회")
# Procedural dates, not punishments: 확정 "(judgment) became final", 가석방 "parole",
# 석방 "release", 종료 "completion", 경과 "elapse", 만료 "expiry"
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
            if last_alcohol:      # 같은 죄 ("the same offense"), or no charge stated at all,
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
