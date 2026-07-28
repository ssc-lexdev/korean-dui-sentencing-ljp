# Few-shot open-book prompt, statute, and sentencing guideline

This file documents, in Korean and in English translation, the full prompt used in the *few-shot
open-book* configuration of Stage 6 (`stage6_llm.py`), for both GPT-5 and Qwen3-8B. The open-book
system prompt is the instruction prompt plus the statutory penalties and the sentencing-guideline
table. The zero-shot closed-book configuration uses **only** the instruction prompt (Section 1
below) — no statute, no guideline, no examples.

> **Which file is authoritative.** Section 1 is verbatim. Sections 2 and 3 are a *lightly condensed
> reading version* of the statute and guideline: amendment annotations are dropped, Criminal Act
> Articles 62-2 to 65 are summarised in one line, and the guideline table is reformatted. The string
> the models actually received is `2_experiment/2-1_experiment_code/law_context.txt`, which `stage6_llm.py` reads directly.
> Build the prompt from that file, not from the blocks below.

> The English text is an unofficial working translation provided for readers; the Korean is
> authoritative. Case facts in the few-shot examples come from the LBOX OPEN corpus; the examples are
> identified by their case `id` and are fully reproducible from the deterministic rule in Section 4.

---

## 1. Instruction (system) prompt — `SYS_BASE`

### Korean (as sent)
```
당신은 한국 음주운전 형사사건의 양형을 예측하는 AI입니다. 주어진 범죄사실만 보고 법원이 선고할 징역 레벨을 예측하세요.

[징역 레벨 정의]
0: 벌금형 또는 징역형 미선고
1: 단기 징역 (약 3~10개월)
2: 중기 징역 (약 12~17개월)
3: 장기 징역 (약 18개월 이상)

반드시 아래 형식으로만 답하세요. 다른 말은 쓰지 마세요.
class: <0,1,2,3 중 하나의 정수>
factors: <판단 근거가 된 주요 양형인자(예: 음주 전과, 혈중알코올농도, 운전거리, 범행연도, 차종)>
```

### English translation
```
You are an AI that predicts the sentence in Korean drunk-driving criminal cases. Based only on the
given criminal facts, predict the imprisonment level the court will impose.

[Imprisonment-level definitions]
0: a fine, or no custodial sentence imposed
1: short imprisonment (about 3-10 months)
2: medium imprisonment (about 12-17 months)
3: long imprisonment (about 18 months or more)

Answer ONLY in the following format. Do not write anything else.
class: <an integer, one of 0,1,2,3>
factors: <the main sentencing factors behind your judgment (e.g., prior DUI record, blood-alcohol
concentration, driving distance, year of offence, vehicle type)>
```

For the open-book configuration, the heading `[관련 법령 및 양형기준]` ("[Relevant statutes and
sentencing guideline]") and Sections 2–3 below are appended to `SYS_BASE` to form `SYS_OPEN`.

---

## 2. Statutory penalties (법정형) — open-book context, part 1

### Korean (as sent)
```
<음주운전 법정형>
형법
제148조의2(벌칙) ① 제44조제1항, 제2항 또는 제5항을 위반(자동차등 또는 노면전차를 운전한 경우로 한정한다. 다만, 개인형 이동장치를 운전한 경우는 제외한다. 이하 이 조에서 같다)하여 벌금 이상의 형을 선고받고 그 형이 확정된 날부터 10년 내에 다시 같은 조 제1항, 제2항 또는 제5항을 위반한 사람(형이 실효된 사람도 포함한다)은 다음 각 호의 구분에 따라 처벌한다.
1. 제44조제2항 또는 제5항을 위반한 사람은 1년 이상 6년 이하의 징역이나 500만원 이상 3천만원 이하의 벌금에 처한다.
2. 제44조제1항을 위반한 사람 중 혈중알코올농도가 0.2퍼센트 이상인 사람은 2년 이상 6년 이하의 징역이나 1천만원 이상 3천만원 이하의 벌금에 처한다.
3. 제44조제1항을 위반한 사람 중 혈중알코올농도가 0.03퍼센트 이상 0.2퍼센트 미만인 사람은 1년 이상 5년 이하의 징역이나 500만원 이상 2천만원 이하의 벌금에 처한다.
② 다음 각 호의 어느 하나에 해당하는 사람은 1년 이상 5년 이하의 징역이나 500만원 이상 2천만원 이하의 벌금에 처한다.
1. 술에 취한 상태에 있다고 인정할 만한 상당한 이유가 있는 사람으로서 제44조제2항에 따른 경찰공무원의 측정에 응하지 아니하는 사람(자동차등 또는 노면전차를 운전한 경우로 한정한다)
2. 술에 취한 상태에 있다고 인정할 만한 상당한 이유가 있는 사람으로서 제44조제5항을 위반하여 자동차등 또는 노면전차를 운전한 후 음주측정방해행위를 한 사람
③ 제44조제1항을 위반하여 술에 취한 상태에서 자동차등 또는 노면전차를 운전한 사람은 다음 각 호의 구분에 따라 처벌한다.
1. 혈중알코올농도가 0.2퍼센트 이상인 사람은 2년 이상 5년 이하의 징역이나 1천만원 이상 2천만원 이하의 벌금
2. 혈중알코올농도가 0.08퍼센트 이상 0.2퍼센트 미만인 사람은 1년 이상 2년 이하의 징역이나 500만원 이상 1천만원 이하의 벌금
3. 혈중알코올농도가 0.03퍼센트 이상 0.08퍼센트 미만인 사람은 1년 이하의 징역이나 500만원 이하의 벌금
④ 제45조를 위반하여 약물로 인하여 정상적으로 운전하지 못할 우려가 있는 상태에서 자동차등 또는 노면전차를 운전한 사람은 3년 이하의 징역이나 1천만원 이하의 벌금에 처한다.
[전문개정 2018. 12. 24.]
(헌법재판소가 2021. 11. 25., 2022. 5. 26., 2022. 8. 31. 위헌 결정한 제1항을 2023. 1. 3. 법률 제19158호로 개정)

<집행유예 기준(형법)>
제62조(집행유예의 요건) ① 3년 이하의 징역이나 금고 또는 500만원 이하의 벌금의 형을 선고할 경우에 제51조의 사항을 참작하여 그 정상에 참작할 만한 사유가 있는 때에는 1년 이상 5년 이하의 기간 형의 집행을 유예할 수 있다. 다만, 금고 이상의 형을 선고한 판결이 확정된 때부터 그 집행을 종료하거나 면제된 후 3년까지의 기간에 범한 죄에 대하여 형을 선고하는 경우에는 그러하지 아니하다.
② 형을 병과할 경우에는 그 형의 일부에 대하여 집행을 유예할 수 있다.
제62조의2 보호관찰ㆍ사회봉사ㆍ수강명령; 제63조 집행유예의 실효; 제64조 집행유예의 취소; 제65조 집행유예의 효과.
제51조(양형의 조건) 1. 범인의 연령, 성행, 지능과 환경 2. 피해자에 대한 관계 3. 범행의 동기, 수단과 결과 4. 범행 후의 정황.
```

### English translation
```
<Statutory penalties for drunk driving>
Criminal Act / Road Traffic Act
Article 148-2 (Penalties) (1) A person who, having violated Article 44(1), (2), or (5) (limited to
operating a motor vehicle etc. or a tram; operating a personal mobility device is excluded; the same
applies throughout this Article) and having been sentenced to a fine or heavier penalty that became
final, again violates Article 44(1), (2), or (5) within 10 years of the date the sentence became
final (including a person whose sentence has lapsed), shall be punished as classified below.
1. A person who violated Article 44(2) or (5): imprisonment for 1-6 years, or a fine of KRW 5-30M.
2. Among violators of Article 44(1), a person with blood-alcohol concentration (BAC) >= 0.2%:
   imprisonment for 2-6 years, or a fine of KRW 10-30M.
3. Among violators of Article 44(1), a person with BAC >= 0.03% and < 0.2%: imprisonment for
   1-5 years, or a fine of KRW 5-20M.
(2) Any of the following shall be punished by imprisonment for 1-5 years or a fine of KRW 5-20M:
1. a person with substantial reason to be deemed intoxicated who does not comply with a police
   officer's measurement under Article 44(2) (limited to operating a vehicle or tram);
2. a person with substantial reason to be deemed intoxicated who, in violation of Article 44(5),
   after operating a vehicle or tram, obstructs a breath-alcohol test.
(3) A person who, in violation of Article 44(1), operated a vehicle or tram while intoxicated:
1. BAC >= 0.2%: imprisonment for 2-5 years, or a fine of KRW 10-20M;
2. BAC >= 0.08% and < 0.2%: imprisonment for 1-2 years, or a fine of KRW 5-10M;
3. BAC >= 0.03% and < 0.08%: imprisonment for up to 1 year, or a fine of up to KRW 5M.
(4) A person who, in violation of Article 45, operated a vehicle or tram while likely unable to drive
normally due to drugs: imprisonment for up to 3 years or a fine of up to KRW 10M.
[Wholly amended 2018-12-24]
(Paragraph (1), held unconstitutional by the Constitutional Court on 2021-11-25, 2022-05-26 and
2022-08-31, was amended by Act No. 19158 on 2023-01-03.)

<Suspended-sentence provisions (Criminal Act)>
Article 62 (Requirements) (1) When sentencing imprisonment (with or without labour) of up to 3 years
or a fine of up to KRW 5M, the court may suspend execution for 1-5 years if, considering the matters
in Article 51, there are grounds warranting leniency -- except for a crime committed within 3 years
after completion or exemption of execution of a final judgment of imprisonment without labour or
heavier. (2) Where penalties are concurrent, part may be suspended.
Article 62-2 probation / community service / attendance orders; Article 63 invalidation; Article 64
revocation; Article 65 effect of suspension.
Article 51 (Conditions of sentencing): 1. the offender's age, character, intelligence and
environment; 2. relationship to the victim; 3. motive, means and result of the crime; 4. circumstances
after the crime.
```

---

## 3. Sentencing guideline (양형기준표) — open-book context, part 2

### Korean (as sent)
```
음주·무면허운전 양형기준표
▷ 1) 1·2유형 감경영역: 3회 이상 벌금형(집행유예 포함) 이상 동종 전과(5년 이내)가 있으면 징역형 선택 가능.
▷ 2) 1·2유형 기본영역, 3~5유형 감경·기본영역: 위 동종 전과가 있으면 징역형 권고.
▷ 3) 1~3유형 가중영역: ① 위 동종 전과가 있으면 징역형 권고. ② 특별가중인자만 2개 이상이거나 특별가중이 특별감경보다 2개 이상 많으면 징역형 권고.
▷ 4) 4·5유형 가중영역: 동종 전과가 없으면 벌금형 선택 가능(4유형 1,500~2,000만원, 5유형 1,300~2,000만원). 단, 특별가중인자 조건 충족 시 징역형 권고.
▷ 만취상태 처리: ① 고의·예견·면책목적의 자의적 만취는 일반가중인자. ② 해악 소질이 있는 경우 만취를 감경인자로 반영하지 않음. ③ 심신미약에 이르지 않으면 감경인자 아님.

유형 | 구분 | 감경 | 기본 | 가중
1 | 무면허운전 | 50만~150만원 | ~8월 / 100만~200만원 | 6~10월 / 150만~300만원
2 | 음주운전(BAC 0.03~0.08% 미만) | 100만~300만원 | ~8월 / 200만~400만원 | 6~10월 / 300만~500만원
3 | 음주운전(BAC 0.08~0.2% 미만) | 6~10월 / 300만~600만원 | 8월~1년4월 / 500만~800만원 | 1년~1년10월 / 700만~1,000만원
4 | 음주운전(BAC 0.2% 이상) | 1~2년 / 700만~1,200만원 | 1년6월~3년 / 1,000만~1,700만원 | 2년6월~4년
5 | 음주측정거부 | 6월~1년2월 / 300만~1,000만원 | 8월~2년 / 700만~1,500만원 | 1년6월~4년

특별양형인자(행위) 감경: 범행동기 특히 참작 / 도로교통 위험 매우 낮음. 가중: 위험 매우 높음 / 공무수행 지장(5유형).
특별양형인자(행위자) 감경: 청각·언어 장애 / 심신미약(본인 책임 없음) / 자수. 가중: 동종 누범.
일반양형인자(행위) 감경: 생계형 범죄(1유형).
일반양형인자(행위자) 감경: 진지한 반성 / 형사처벌 전력 없음. 가중: 범행 후 증거은폐(시도) / 이종 누범 또는 누범 아닌 동종 전과(10년 미만).
```

### English translation
```
Sentencing Guideline Table for Drunk / Unlicensed Driving (Korean Sentencing Commission)
Note 1) Types 1-2 mitigated range: if there are 3+ same-kind priors of a fine (incl. suspended) or
        heavier within 5 years, imprisonment may be selected.
Note 2) Types 1-2 basic range and Types 3-5 mitigated/basic ranges: with such priors, imprisonment is
        recommended.
Note 3) Types 1-3 aggravated range: (i) with such priors, imprisonment is recommended; (ii) if there
        are 2+ special aggravating factors only, or they exceed special mitigating factors by 2+,
        imprisonment is recommended.
Note 4) Types 4-5 aggravated range: with no same-kind prior, a fine may be selected (Type 4:
        KRW 15-20M; Type 5: KRW 13-20M); but if the special-aggravator condition is met, imprisonment
        is recommended.
Heavy-intoxication handling: (i) voluntary heavy intoxication with intent/foresight/defence purpose
        is a general aggravating factor; (ii) where the offender has a propensity to harm, heavy
        intoxication is not credited as mitigating; (iii) if it does not amount to diminished mental
        capacity, it is not a mitigating factor.

Type | Category | Mitigated | Basic | Aggravated
1 | Unlicensed driving | KRW 0.5-1.5M | up to 8 mo / KRW 1-2M | 6-10 mo / KRW 1.5-3M
2 | DUI (BAC 0.03-<0.08%) | KRW 1-3M | up to 8 mo / KRW 2-4M | 6-10 mo / KRW 3-5M
3 | DUI (BAC 0.08-<0.2%) | 6-10 mo / KRW 3-6M | 8 mo-1 yr 4 mo / KRW 5-8M | 1 yr-1 yr 10 mo / KRW 7-10M
4 | DUI (BAC >= 0.2%) | 1-2 yr / KRW 7-12M | 1 yr 6 mo-3 yr / KRW 10-17M | 2 yr 6 mo-4 yr
5 | Refusal of breath test | 6 mo-1 yr 2 mo / KRW 3-10M | 8 mo-2 yr / KRW 7-15M | 1 yr 6 mo-4 yr

Special factors (conduct) -- mitigating: specially considerable motive; very low road-traffic danger.
  aggravating: very high danger; substantial obstruction of public duty (Type 5).
Special factors (offender) -- mitigating: hearing/speech disability; diminished capacity (not the
  offender's fault); voluntary surrender. aggravating: same-kind repeat offence.
General factors (conduct) -- mitigating: subsistence crime (Type 1).
General factors (offender) -- mitigating: sincere repentance; no prior criminal punishment.
  aggravating: concealment (or attempt) of evidence after the crime; different-kind repeat offence,
  or a same-kind prior not amounting to repeat offence (within 10 years).
```

---

## 4. Few-shot examples (open-book only)

Four class-balanced examples are prepended (after the system prompt), one per ordinal class, selected
**deterministically** as the first case of each class in the **train split** of the verified table.
Each example is a user turn `"[범죄사실]\n<facts>"` followed by an assistant turn
`"class: <c>\nfactors: <factor summary>"`. The factor summary is generated deterministically as
`음주 전과 {prior_dui_count}회, 혈중알코올농도 {bac}%, 운전거리 {distance_km}km, 범행연도 {offense_year}`
("prior DUI record N times, BAC X%, driving distance Y km, year of offence Z").

The `<facts>` are the LBOX OPEN records for the case `id`s below (not redistributed here):

| class | LBOX case id | assistant target — `factors` (Korean) |
|---|---|---|
| 0 | 4867 | 음주 전과 0회, 혈중알코올농도 0.048%, 운전거리 10.0km, 범행연도 2020 |
| 1 | 4781 | 음주 전과 0회, 혈중알코올농도 0.145%, 운전거리 1.0km, 범행연도 2021 |
| 2 | 4525 | 음주 전과 1회, 혈중알코올농도 0.252%, 운전거리 6.0km, 범행연도 2020 |
| 3 | 4575 | 음주 전과 1회, 혈중알코올농도 0.106%, 운전거리 2.0km, 범행연도 2020 |

---

## 5. Message assembly and decoding

- **Open-book system prompt** = `SYS_BASE` (Section 1) + `\n\n[관련 법령 및 양형기준]\n` + statute
  (Section 2) + sentencing guideline (Section 3).
- **Message list** = `[system]` + the four example user/assistant pairs (Section 4) + the target
  `[범죄사실]\n<facts>` user turn.
- **Zero-shot closed-book** uses only `SYS_BASE` as the system prompt, with **no** statute, guideline,
  or examples.
- **Decoding**: GPT-5 — `reasoning_effort = "minimal"`, `max_completion_tokens = 1200`; Qwen3-8B —
  `temperature = 0.0`, `max_tokens = 300`. Each unique case is predicted once and cached
  (`outputs/llm_cache_<model>_<variant>.json`). Source: `stage6_llm.py`.
