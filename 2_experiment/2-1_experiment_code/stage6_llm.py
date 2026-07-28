"""
Stage 6 - LLM inference (research plan v4, Section 4.2; H2 and rationales for H3/Stage 8).

Runs GPT-5 (OpenAI) and Qwen3-8B (OpenRouter) in two settings each:
  - zero-shot, closed-book  (no statute context)
  - few-shot,  open-book    (statute + sentencing guideline + 4 class-balanced examples)
LLMs do not train, so each unique case is predicted once and reused for every CV test fold
in Stage 7. Each call returns a 4-class label and the sentencing factors it relied on
(for the Stage 8 ML<->LLM agreement). Results are cached so the run is resumable.

Env: OPENAI_API_KEY, OPENROUTER_API_KEY
Args: optional integer = number of cases (for a small test run); default = all 1,500.
Outputs: outputs/llm_cache_<model>_<variant>.json, outputs/stage6_llm_predictions.parquet
"""
import sys, io, os, re, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
from openai import OpenAI

import config
import features as F
config.set_seed()
OUT = config.OUTPUT_DIR

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None

# -- Statute / guideline (open-book context) ------------------------------
# The original run read this text out of two .docx files. Those files are not
# part of the release, so the exact string they produced is shipped instead, as
# law_context.txt, next to this script. It is byte-identical to what was sent.
# 5_supplement/few_shot_open_book_prompt.md renders the same material for human
# reading and is lightly condensed; do not build the prompt from it.
LAW = config.LAW_CONTEXT.read_text(encoding="utf-8").rstrip("\n")

# -- Data ------------------------------------------------------------------
v = pd.read_parquet(OUT / "stage2_features_verified.parquet")
prep = (pd.read_parquet(OUT / "stage1_prepared.parquet")
        .drop_duplicates("id")[["id", "facts"]])   # split already in verified table
df = v.merge(prep, on="id", how="left").reset_index(drop=True)
if LIMIT:
    df = df.groupby("class4", group_keys=False).head(max(2, LIMIT // 4)).head(LIMIT)
print(f"Cases to run: {len(df)}")

CLASS_DEF = ("0: 벌금형 또는 징역형 미선고\n"
             "1: 단기 징역 (약 3~10개월)\n"
             "2: 중기 징역 (약 12~17개월)\n"
             "3: 장기 징역 (약 18개월 이상)")
SYS_BASE = ("당신은 한국 음주운전 형사사건의 양형을 예측하는 AI입니다. 주어진 범죄사실만 보고 "
            "법원이 선고할 징역 레벨을 예측하세요.\n\n[징역 레벨 정의]\n" + CLASS_DEF +
            "\n\n반드시 아래 형식으로만 답하세요. 다른 말은 쓰지 마세요.\n"
            "class: <0,1,2,3 중 하나의 정수>\n"
            "factors: <판단 근거가 된 주요 양형인자(예: 음주 전과, 혈중알코올농도, 운전거리, 범행연도, 차종)>")
SYS_OPEN = SYS_BASE + "\n\n[관련 법령 및 양형기준]\n" + LAW

# -- Few-shot examples (one per class, from the train split) --------------
train = df if LIMIT else df[df["split"] == "train"]
fewshot = []
for c in range(config.N_CLASSES):
    sub = train[train["class4"] == c]
    if len(sub) == 0:
        continue
    ex = sub.iloc[0]
    fa = (f"음주 전과 {ex['prior_dui_count']}회, 혈중알코올농도 {ex['bac']}%, "
          f"운전거리 {ex['distance_km']}km, 범행연도 {ex['offense_year']}")
    fewshot.append({"role": "user", "content": f"[범죄사실]\n{ex['facts']}"})
    fewshot.append({"role": "assistant", "content": f"class: {c}\nfactors: {fa}"})
EX_IDS = set()  # exclude few-shot example ids from eval if needed (small, ignored here)

# -- Clients ---------------------------------------------------------------
gpt = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
qwen = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
              base_url="https://openrouter.ai/api/v1")

CONFIGS = [
    ("gpt-5",            "gpt",  "zero_shot_closed_book", SYS_BASE, []),
    ("gpt-5",            "gpt",  "few_shot_open_book",    SYS_OPEN, fewshot),
    ("qwen/qwen3-8b",    "qwen", "zero_shot_closed_book", SYS_BASE, []),
    ("qwen/qwen3-8b",    "qwen", "few_shot_open_book",    SYS_OPEN, fewshot),
]

def parse(txt):
    m = re.search(r"class\s*[:=]?\s*([0-3])", txt)
    cls = int(m.group(1)) if m else -1
    fm = re.search(r"factors\s*[:=]?\s*(.+)", txt, re.S)
    return cls, (fm.group(1).strip()[:300] if fm else "")

def call(client, model, system, shots, facts):
    msgs = [{"role": "system", "content": system}] + shots + \
           [{"role": "user", "content": f"[범죄사실]\n{facts}"}]
    kw = dict(model=model, messages=msgs)
    # gpt-5 is a reasoning model: minimal effort keeps the answer from being truncated by
    # reasoning tokens (the task is a short classification), and cuts cost sharply.
    if model.startswith("gpt-5"):
        kw["max_completion_tokens"] = 1200
        kw["reasoning_effort"] = "minimal"
    else:
        kw["max_tokens"] = 300
        kw["temperature"] = 0.0
    r = client.chat.completions.create(**kw)
    return r.choices[0].message.content or "", r.usage

rows = []
for model, who, variant, system, shots in CONFIGS:
    client = gpt if who == "gpt" else qwen
    cache_path = OUT / f"llm_cache_{who}_{variant}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    tok_in = tok_out = 0
    print(f"\n=== {model} | {variant} ===")
    for i, row in df.iterrows():
        cid = str(int(row["id"]))
        if cid in cache:
            raw = cache[cid]["raw"]
        else:
            for attempt in range(3):
                try:
                    raw, usage = call(client, model, system, shots, row["facts"])
                    tok_in += usage.prompt_tokens; tok_out += usage.completion_tokens
                    cache[cid] = {"raw": raw}
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                    break
                except Exception as e:
                    print(f"  err id={cid} ({attempt+1}/3): {str(e)[:120]}")
                    time.sleep(3)
            else:
                raw = ""
        cls, fac = parse(raw)
        rows.append({"id": int(row["id"]), "model": who, "variant": variant,
                     "y_true": int(row["class4"]), "y_pred": cls, "factors": fac})
        if (df.index.get_loc(i) + 1) % 25 == 0:
            print(f"  {df.index.get_loc(i)+1}/{len(df)}  (tok in {tok_in}, out {tok_out})")
    print(f"  done. tokens in={tok_in} out={tok_out}")

pred = pd.DataFrame(rows)
pred.to_parquet(OUT / "stage6_llm_predictions.parquet", index=False)
print("\n=== Parse success & quick accuracy ===")
for (who, variant), g in pred.groupby(["model", "variant"]):
    ok = (g["y_pred"] >= 0).mean()
    acc = (g["y_pred"] == g["y_true"]).mean()
    print(f"  {who:5s} {variant:22s} parsed={ok*100:.0f}%  raw_acc={acc*100:.1f}%")
print("\nStage 6 done.")
