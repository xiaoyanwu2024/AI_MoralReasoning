# ---------------------------------------------------------------------------
# Author: Xiaoyan Wu
# Date: June 2026
# Contact: xiaoyan.psych@gmail.com
# ---------------------------------------------------------------------------

"""
Gemini data-collection script: runs Google Gemini 2.5 Pro on the Firsthand-Perspective materials.

Like the DeepSeek and Cerebras(gpt-oss) scripts, this imports the scenario materials, scale
items, persona sampling, build_system_prompt(), and build_user_prompt() directly from
firsthand.py -- the prompt content is identical across all three models; only the API target
changes.

Technical detail: Gemini exposes an OpenAI-compatible endpoint
(generativelanguage.googleapis.com/v1beta/openai/), called here with the same openai.OpenAI
client used by the DeepSeek/Cerebras scripts, keeping the code structure consistent and making it
easy to verify that the prompt-construction logic is identical across all three (there is only one
copy of build_system_prompt/build_user_prompt).

Gemini 2.5 Pro has a "thinking" mechanism that, like gpt-oss-120b, consumes a hidden token budget
before producing visible output, so max_tokens is likewise raised, and an empty response on
failure is treated as an error rather than failing silently.

Usage:
    pip install openai
    export GEMINI_API_KEY=your_key
    python firsthand_gemini.py --n_per_condition 200
"""

import argparse
import csv
import json
import random
import re
import time

from firsthand import (
    CONDITION_ENDINGS,
    MOTIVE_ITEMS,
    build_scenario_text,
    build_system_prompt,
    build_user_prompt,
    sample_persona,
    sample_scenario_genders,
    CHARACTER_ITEMS,
    QuotaExhaustedError,
    is_hard_error,
)

META_FIELDS = ["condition", "model", "persona_age", "persona_gender", "persona_is_student",
               "schmitt_gender", "bauer_gender"]
FIELDNAMES = META_FIELDS + list(MOTIVE_ITEMS.keys()) + list(CHARACTER_ITEMS)

# Gemini's paid tier has much looser rate limits than Cerebras's free tier, but we still start conservatively to avoid bursts of 429s
MIN_INTERVAL_SECONDS = 1.0

_GEMINI_CLIENT = None


def call_model(system_prompt: str, user_prompt: str, model: str, temperature: float = 0.8,
                max_retries: int = 2, max_tokens: int = 4000) -> str:
    global _GEMINI_CLIENT
    import os
    from openai import OpenAI

    if _GEMINI_CLIENT is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Environment variable GEMINI_API_KEY not found; please run export GEMINI_API_KEY=your_key first")
        _GEMINI_CLIENT = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = _GEMINI_CLIENT.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError(
                    f"Model returned empty content (likely the thinking-token budget exhausted max_tokens, "
                    f"finish_reason={resp.choices[0].finish_reason}）"
                )
            return content
        except Exception as e:
            if is_hard_error(e):
                raise QuotaExhaustedError(str(e)) from e
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Call failed after {max_retries} retries: {last_err}")


def parse_json_response(raw_text: str) -> dict:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON in the model output: {raw_text[:200]}")
    return json.loads(match.group(0))


def _count_existing_by_condition(output_csv: str) -> dict:
    import os
    counts = {c: 0 for c in CONDITION_ENDINGS.keys()}
    if not os.path.exists(output_csv):
        return counts
    with open(output_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cond = row.get("condition")
            if cond in counts:
                counts[cond] += 1
    return counts


def run_collection(n_per_condition: int, model: str, output_csv: str, seed: int = 42, max_tokens: int = 4000):
    """Incremental CSV writing + resume support + fixed fieldnames -- the same robustness mechanism as firsthand_cerebras_llama.py."""
    import os
    rng = random.Random(seed)
    conditions = list(CONDITION_ENDINGS.keys())

    existing_counts = _count_existing_by_condition(output_csv)
    remaining = {c: max(0, n_per_condition - existing_counts[c]) for c in conditions}
    if any(existing_counts[c] > 0 for c in conditions):
        print(f"Existing progress detected: {existing_counts}; this run will only fill the remaining gap: {remaining}")

    file_exists = os.path.exists(output_csv) and sum(existing_counts.values()) > 0
    f_out = open(output_csv, "a" if file_exists else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES, extrasaction="ignore")
    if not file_exists:
        writer.writeheader()
        f_out.flush()
    n_done, n_failed = 0, 0
    quota_exhausted = False

    try:
        for condition in conditions:
            if quota_exhausted:
                break
            for i in range(remaining[condition]):
                call_start = time.monotonic()
                persona = sample_persona(rng)
                genders = sample_scenario_genders(rng)
                scenario_text = build_scenario_text(condition, genders)
                system_prompt = build_system_prompt(persona)
                user_prompt = build_user_prompt(scenario_text, rng)

                try:
                    raw = call_model(system_prompt, user_prompt, model, max_tokens=max_tokens)
                    parsed = parse_json_response(raw)
                except QuotaExhaustedError as e:
                    n_failed += 1
                    print(f"[condition={condition} attempt {i+1}] quota/balance exhausted, stopping this run: {e}")
                    quota_exhausted = True
                    break
                except Exception as e:
                    n_failed += 1
                    print(f"[condition={condition} attempt {i+1}] failed: {e}")
                else:
                    row = {
                        "condition": condition,
                        "model": f"{model}(Gemini)",
                        "persona_age": persona["age"],
                        "persona_gender": persona["gender"],
                        "persona_is_student": persona["is_student"],
                        "schmitt_gender": genders["schmitt_gender"],
                        "bauer_gender": genders["bauer_gender"],
                    }
                    row.update(parsed.get("motive_items", {}))
                    row.update(parsed.get("character_items", {}))
                    writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
                    f_out.flush()
                    n_done += 1
                    print(f"[condition={condition} this-run {i+1}/{remaining[condition]}, overall {existing_counts[condition]+i+1}/{n_per_condition}] "
                          f"done ({n_done} rows saved so far, {n_failed} failed)")

                elapsed = time.monotonic() - call_start
                if elapsed < MIN_INTERVAL_SECONDS:
                    time.sleep(MIN_INTERVAL_SECONDS - elapsed)
    finally:
        f_out.close()

    if quota_exhausted:
        print(f"\nRun terminated early due to quota/balance exhaustion; saved {n_done} new rows to {output_csv} ({n_failed} failed)")
    else:
        print(f"\nSaved {n_done} rows to {output_csv} ({n_failed} failed)")


def quick_sanity_check(output_csv: str):
    import statistics

    with open(output_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    dims = ["deontic", "prosocial", "individualistic", "competitive"]
    print("\n=== Mean rating by condition x motive dimension (Gemini) ===")
    for condition in ["no_wb", "internal_wb", "external_wb", "public_wb"]:
        cond_rows = [r for r in rows if r["condition"] == condition]
        if not cond_rows:
            continue
        means = {}
        for dim in dims:
            keys = [k for k in cond_rows[0] if k.startswith(dim)]
            vals = [float(r[k]) for r in cond_rows for k in keys if r.get(k)]
            means[dim] = round(statistics.mean(vals), 2) if vals else None
        char_vals = [float(r[c]) for r in cond_rows for c in CHARACTER_ITEMS if r.get(c)]
        char_mean = round(statistics.mean(char_vals), 2) if char_vals else None
        print(f"{condition:12s} -> motives={means}, character_judgment_mean={char_mean}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_condition", type=int, default=200)
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--output", type=str, default="firsthand_full_gemini25pro.csv")
    parser.add_argument("--max_tokens", type=int, default=4000)
    args = parser.parse_args()

    run_collection(args.n_per_condition, args.model, args.output, max_tokens=args.max_tokens)
    quick_sanity_check(args.output)
