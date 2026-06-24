# ---------------------------------------------------------------------------
# Author: Xiaoyan Wu
# Date: June 2026
# Contact: xiaoyan.psych@gmail.com
# ---------------------------------------------------------------------------

"""
Hugging Face free Inference API data-collection script: runs Llama-3.1-8B-Instruct on the
Secondhand-Perspective materials (the "heard about it afterward" framing).

Key differences from firsthand_hf_llama.py:
- Different scenario framing: the respondent "happened to hear about" what the team did, rather
  than witnessing it directly.
- Personas are sampled to match the real demographic composition of the Secondhand-Perspective
  sample: age ~ N(49.43, 15.06), gender 64.3% female / 34.8% male / 0.9% other, 5.66% students.
- Scenario materials and persona sampling are imported from secondhand.py; scale items and
  build_user_prompt() are imported from firsthand.py (the same reuse pattern as
  secondhand_cerebras_llama.py's reuse for Cerebras, just with call_model swapped for the HF
  endpoint).
- Same incremental writing + resume support + hard-error (quota/balance exhausted) detection as
  firsthand_hf_llama.py.

Usage:
    pip install huggingface_hub
    export HF_TOKEN=your_key
    python secondhand_hf_llama.py --n_per_condition 200
"""

import argparse
import csv
import json
import random
import re
import time

from firsthand import (
    MOTIVE_ITEMS,
    build_user_prompt,
    CHARACTER_ITEMS,
    QuotaExhaustedError,
    is_hard_error,
)
from secondhand import (
    CONDITION_ENDINGS,
    build_scenario_text,
    build_system_prompt,
    sample_persona,
    sample_scenario_genders,
)

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Fieldnames are fixed in advance rather than inferred from the first successful record at runtime
META_FIELDS = ["condition", "model", "persona_age", "persona_gender", "persona_is_student",
               "schmitt_gender", "bauer_gender"]
FIELDNAMES = META_FIELDS + list(MOTIVE_ITEMS.keys()) + list(CHARACTER_ITEMS)

# Queueing is common on HF's free-tier serverless inference; starting conservatively
MIN_INTERVAL_SECONDS = 1.5

_CLIENT = None


def call_model(system_prompt: str, user_prompt: str, temperature: float = 0.8,
                max_retries: int = 3, max_tokens: int = 600) -> str:
    global _CLIENT
    from huggingface_hub import InferenceClient

    if _CLIENT is None:
        import os
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Environment variable HF_TOKEN not found; please run export HF_TOKEN=your_key first")
        _CLIENT = InferenceClient(model=MODEL_ID, token=token)

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = _CLIENT.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("Model returned empty content")
            return content
        except Exception as e:
            if is_hard_error(e):
                raise QuotaExhaustedError(str(e)) from e
            last_err = e
            time.sleep(2 * (attempt + 1))
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


def run_collection(n_per_condition: int, output_csv: str, seed: int = 42, max_tokens: int = 600):
    """Incremental CSV writing + resume support + hard-error detection -- identical to firsthand_hf_llama.py."""
    rng = random.Random(seed)
    conditions = list(CONDITION_ENDINGS.keys())

    existing_counts = _count_existing_by_condition(output_csv)
    remaining = {c: max(0, n_per_condition - existing_counts[c]) for c in conditions}
    if any(existing_counts[c] > 0 for c in conditions):
        print(f"Existing progress detected: {existing_counts}; this run will only fill the remaining gap: {remaining}")

    import os
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
                    raw = call_model(system_prompt, user_prompt, max_tokens=max_tokens)
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
                        "model": f"{MODEL_ID}(HF Inference API)",
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
                    print(f"[condition={condition} this-run {i+1}/{remaining[condition]}, overall {existing_counts[condition]+i+1}/{n_per_condition}] done ({n_done} rows saved so far, {n_failed} failed)")

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
    print("\n=== Secondhand-Perspective: mean rating by condition x motive dimension (Llama-3.1-8B-Instruct via HF Inference API) ===")
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
    parser.add_argument("--output", type=str, default="secondhand_full_hf_llama3.1_8b.csv")
    parser.add_argument("--max_tokens", type=int, default=600)
    args = parser.parse_args()

    run_collection(args.n_per_condition, args.output, max_tokens=args.max_tokens)
    quick_sanity_check(args.output)
