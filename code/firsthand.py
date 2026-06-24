# ---------------------------------------------------------------------------
# Author: Xiaoyan Wu
# Date: June 2026
# Contact: xiaoyan.psych@gmail.com
# ---------------------------------------------------------------------------

"""
Data-collection script: LLM simulated respondents vs. Firsthand-Perspective human data
(motive attribution + moral character judgment).

Design (matches the study's analysis plan):
- Uses only the four disclosure conditions of the Firsthand-Perspective materials
  (between-subjects; each API call represents one simulated respondent who has "seen one
  condition").
- Personas are sampled to match the real demographic composition of the Firsthand-Perspective
  human sample: age ~ N(33.69, 11.76), gender 77.6% female / 22.4% male.
- Scenario-character genders (Dr Schmitt x Dr Bauer) are randomized 2x2.
- 12-item motive scale (deontic/prosocial/individualistic/competitive) + 16-item moral
  character scale.
- Item order is shuffled independently for every call.
- Output is required to be strict JSON; the model is not asked to "explain before rating" (to
  avoid anchoring contamination).

Usage:
    pip install openai
    export DEEPSEEK_API_KEY=your_key
    python firsthand.py --n_per_condition 20 --model deepseek-chat
"""

import argparse
import json
import random
import re
import csv
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Hard-error detection: quota/balance-exhausted errors are not worth retrying and should
#    terminate the whole run immediately rather than burning through the remaining retries
#    (lesson learned after a Gemini balance-exhaustion incident on 2026-06-23, where the
#    script kept retrying hundreds of times after the balance ran out). Keywords below are
#    taken from the actual error strings encountered that day.
# ---------------------------------------------------------------------------

HARD_ERROR_KEYWORDS = [
    "prepayment credits are depleted",
    "tokens per day limit exceeded",
    "requests per hour limit exceeded",
    "requests per day",
    "per_day",
    "insufficient_quota",
    "402 payment required",
    "payment required",
    # Authentication errors: an invalid/expired key or insufficient permissions -- retrying
    # the same key is pointless and the run should stop immediately (lesson learned after an
    # expired Gemini key wasn't caught early on the night of 2026-06-23).
    "invalid authentication credentials",
    "unauthenticated",
    "invalid api key",
    "incorrect api key",
]


class QuotaExhaustedError(RuntimeError):
    """Quota/balance exhausted -- retrying will not succeed; the caller should stop the run."""


def is_hard_error(err: Exception) -> bool:
    err_str = str(err).lower()
    return any(kw in err_str for kw in HARD_ERROR_KEYWORDS)

# ---------------------------------------------------------------------------
# 1. Scenario materials: Firsthand-Perspective, 4 conditions (adapted from the OSF
#    "Study Material" PDF for the firsthand-perspective version)
#
# LICENSE NOTE: SCENARIO_INTRO and CONDITION_ENDINGS below reproduce text from the original
# study materials of Brotzeller, van Houwelingen, Gollwitzer, & Fischer (2025), OSF: q4rsn,
# which are licensed CC BY-NC-ND 4.0 (Attribution-NonCommercial-NoDerivatives;
# https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode). This text is NOT covered by this
# repository's MIT license -- it remains governed by the original CC BY-NC-ND 4.0 terms
# (noncommercial use only, no derivative materials), with attribution to the original authors.
# All other code in this repository (everything outside this text block) is MIT-licensed.
# ---------------------------------------------------------------------------

SCENARIO_INTRO = """You are a physician at the hospital "Klinik am See". You have already been \
working with your team in this hospital for several years. Together with your team, you advise \
people who are ill on a daily basis, make diagnoses and carry out treatments.

The other day {perpetrator_ref} from your team, Dr Schmitt, billed the health insurance company \
for more services than were actually provided for some patients. For example, services were \
billed by Dr Schmitt without {perpetrator_pronoun} even being present during the billed treatments."""

CONDITION_ENDINGS = {
    "no_wb": (
        "When {wb_ref} of your colleagues, Dr Bauer, learned about it, "
        "{wb_pronoun} did not pass on any information about it."
    ),
    "internal_wb": (
        "When {wb_ref} of your colleagues, Dr Bauer, learned about this, {wb_pronoun} passed on "
        "information about it to the hospital's compliance department, which is responsible for "
        "ensuring that the hospital complies with all legal requirements."
    ),
    "external_wb": (
        "When {wb_ref} of your colleagues, Dr Bauer, learned about this, {wb_pronoun} passed on "
        "information about it to the relevant authorities."
    ),
    "public_wb": (
        "When {wb_ref} of your colleagues, Dr Bauer, learned about this, {wb_pronoun} passed on "
        "information about it to a newspaper that is distributed throughout Germany."
    ),
}

# Character-gender randomization: Dr Schmitt (wrongdoer) and Dr Bauer (discloser) are each
# independently randomized male/female, matching the original study's 2x2 counterbalanced design.
GENDER_PRONOUNS = {
    "male": {"ref": "another physician", "pronoun": "him"},
    "female": {"ref": "another physician", "pronoun": "her"},
}
WB_GENDER_PRONOUNS = {
    "male": {"ref": "one", "pronoun": "he"},
    "female": {"ref": "one", "pronoun": "she"},
}

# ---------------------------------------------------------------------------
# 2. Scale items (adapted from the original codebook / the paper's Measures section)
# ---------------------------------------------------------------------------

MOTIVE_ITEMS = {
    "deontic_1": "follow a moral obligation",
    "deontic_2": "act in accordance with moral values",
    "deontic_3": "do the right thing",
    "prosocial_1": "help others",
    "prosocial_2": "support others",
    "prosocial_3": "assist others",
    "individualistic_1": "benefit themselves",
    "individualistic_2": "pursue their own interests",
    "individualistic_3": "gain an advantage for themselves",
    "competitive_1": "harm others",
    "competitive_2": "humiliate others",
    "competitive_3": "make others look bad",
}

CHARACTER_ITEMS = [
    "humble", "kind", "forgiving", "giving", "helpful", "grateful", "empathetic",
    "cooperative", "courageous", "fair", "principled", "responsible", "just",
    "honest", "trustworthy", "loyal",
]

LIKERT_SCALE = "1 = strongly disagree, 2 = disagree, 3 = rather disagree, 4 = rather agree, 5 = agree, 6 = strongly agree"

# ---------------------------------------------------------------------------
# 3. Persona sampling: matched to the real demographic composition of the
#    Firsthand-Perspective human sample (see the paper's Method section)
# ---------------------------------------------------------------------------

def sample_persona(rng: random.Random) -> dict:
    age = round(rng.gauss(33.69, 11.76))
    age = max(18, min(80, age))
    gender = rng.choices(["female", "male"], weights=[77.6, 22.4])[0]  # real Firsthand-Perspective proportions, other=0
    is_student = rng.random() < 0.304
    student_clause = (
        "You are currently a university student. "
        if is_student
        else "You are not currently a student. "
    )
    return {"age": age, "gender": gender, "is_student": is_student, "student_clause": student_clause}


def sample_scenario_genders(rng: random.Random) -> dict:
    schmitt_gender = rng.choice(["male", "female"])
    bauer_gender = rng.choice(["male", "female"])
    return {"schmitt_gender": schmitt_gender, "bauer_gender": bauer_gender}


# ---------------------------------------------------------------------------
# 4. Prompt construction
# ---------------------------------------------------------------------------

def build_scenario_text(condition: str, genders: dict) -> str:
    schmitt = GENDER_PRONOUNS[genders["schmitt_gender"]]
    bauer = WB_GENDER_PRONOUNS[genders["bauer_gender"]]
    intro = SCENARIO_INTRO.format(
        perpetrator_ref=schmitt["ref"], perpetrator_pronoun=schmitt["pronoun"]
    )
    ending = CONDITION_ENDINGS[condition].format(wb_ref=bauer["ref"], wb_pronoun=bauer["pronoun"])
    return intro + "\n\n" + ending


def build_system_prompt(persona: dict) -> str:
    return f"""You are participating in a psychology survey about how people interpret others' \
behavior in workplace scenarios.

You are answering as a {persona['age']}-year-old {persona['gender']} from Germany. \
{persona['student_clause']}You are a general member of the public taking part in an online \
survey, not an expert in ethics, law, or psychology.

Read the scenario carefully, then answer every statement below as this person would, based only \
on the information given. There are no right or wrong answers; give your genuine personal \
impression. Respond ONLY in the JSON format specified at the end. Do not add any explanation \
outside the JSON."""


def build_user_prompt(scenario_text: str, rng: random.Random) -> str:
    motive_keys = list(MOTIVE_ITEMS.keys())
    rng.shuffle(motive_keys)
    character_keys = CHARACTER_ITEMS.copy()
    rng.shuffle(character_keys)

    motive_lines = "\n".join(f'  - "{k}": Dr Bauer wanted to {MOTIVE_ITEMS[k]}.' for k in motive_keys)
    character_lines = "\n".join(f'  - "{k}": Dr Bauer is {k}.' for k in character_keys)

    json_motive_keys = ", ".join(f'"{k}": <1-6>' for k in motive_keys)
    json_character_keys = ", ".join(f'"{k}": <1-6>' for k in character_keys)

    return f"""The scenario:

{scenario_text}

As a reminder: Dr Bauer was the person who learned that Dr Schmitt was billing the insurance \
company for more services than were actually provided.

Please rate the extent to which you agree with each statement, using this scale:
{LIKERT_SCALE}

Motive items (Dr Bauer wanted to...):
{motive_lines}

Character items (Dr Bauer is...):
{character_lines}

Respond ONLY with this exact JSON structure (no extra text):
{{
  "motive_items": {{{json_motive_keys}}},
  "character_items": {{{json_character_keys}}}
}}"""


# ---------------------------------------------------------------------------
# 5. API call (currently: DeepSeek, OpenAI-compatible endpoint; see the commented-out
#    Anthropic version below if you need to switch back)
# ---------------------------------------------------------------------------
#
# Before running, set the environment variable:
#   export DEEPSEEK_API_KEY=your_key
# Install the dependency:
#   pip install openai

_DEEPSEEK_CLIENT = None


def call_model(system_prompt: str, user_prompt: str, model: str, temperature: float = 0.8) -> str:
    global _DEEPSEEK_CLIENT
    import os
    from openai import OpenAI

    if _DEEPSEEK_CLIENT is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Environment variable DEEPSEEK_API_KEY not found; please run "
                "export DEEPSEEK_API_KEY=your_key first"
            )
        _DEEPSEEK_CLIENT = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    resp = _DEEPSEEK_CLIENT.chat.completions.create(
        model=model,
        max_tokens=600,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


# --- To switch back to the Anthropic API, replace call_model above with this version ---
# def call_model(system_prompt, user_prompt, model, temperature=0.8):
#     import anthropic
#     client = anthropic.Anthropic()
#     resp = client.messages.create(
#         model=model, max_tokens=600, temperature=temperature,
#         system=system_prompt,
#         messages=[{"role": "user", "content": user_prompt}],
#     )
#     return resp.content[0].text


def parse_json_response(raw_text: str) -> dict:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON in the model output: {raw_text[:200]}")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# 6. Main loop: call independently n_per_condition times per condition (between-subjects;
#    no persona is reused across calls)
# ---------------------------------------------------------------------------

def run_collection(n_per_condition: int, model: str, output_csv: str, seed: int = 42):
    """Writes the CSV incrementally: each successful row is flushed to disk immediately, so an
    interruption does not lose already-completed data."""
    rng = random.Random(seed)
    conditions = list(CONDITION_ENDINGS.keys())

    fieldnames = None
    f_out = None
    writer = None
    n_done = 0

    try:
        for condition in conditions:
            for i in range(n_per_condition):
                persona = sample_persona(rng)
                genders = sample_scenario_genders(rng)
                scenario_text = build_scenario_text(condition, genders)
                system_prompt = build_system_prompt(persona)
                user_prompt = build_user_prompt(scenario_text, rng)

                try:
                    raw = call_model(system_prompt, user_prompt, model)
                    parsed = parse_json_response(raw)
                except Exception as e:
                    print(f"[condition={condition} attempt {i+1}] call/parse failed: {e}")
                    continue

                row = {
                    "condition": condition,
                    "model": model,
                    "persona_age": persona["age"],
                    "persona_gender": persona["gender"],
                    "persona_is_student": persona["is_student"],
                    "schmitt_gender": genders["schmitt_gender"],
                    "bauer_gender": genders["bauer_gender"],
                }
                row.update(parsed.get("motive_items", {}))
                row.update(parsed.get("character_items", {}))

                if writer is None:
                    fieldnames = sorted(row.keys())
                    f_out = open(output_csv, "w", newline="", encoding="utf-8")
                    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow({k: row.get(k, "") for k in fieldnames})
                f_out.flush()
                n_done += 1
                print(f"[condition={condition} {i+1}/{n_per_condition}] done ({n_done} rows saved so far)")
                time.sleep(0.3)  # avoid rate limiting; adjust to the actual API quota
    finally:
        if f_out:
            f_out.close()

    if n_done == 0:
        print("No successful calls; no output file was generated.")
        return
    print(f"\nSaved {n_done} rows to {output_csv}")


# ---------------------------------------------------------------------------
# 7. Quick sanity check: summarize mean motive ratings by condition and compare the direction
#    against the human data reported in the paper
# ---------------------------------------------------------------------------

def quick_sanity_check(output_csv: str):
    import statistics

    with open(output_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    dims = ["deontic", "prosocial", "individualistic", "competitive"]
    print("\n=== Mean rating by condition x motive dimension (LLM simulated respondents) ===")
    for condition in ["no_wb", "internal_wb", "external_wb", "public_wb"]:
        cond_rows = [r for r in rows if r["condition"] == condition]
        if not cond_rows:
            continue
        means = {}
        for dim in dims:
            keys = [k for k in cond_rows[0] if k.startswith(dim)]
            vals = [float(r[k]) for r in cond_rows for k in keys if r.get(k)]
            means[dim] = round(statistics.mean(vals), 2) if vals else None
        print(f"{condition:12s} -> {means}")

    print("\nReference: expected direction from the Firsthand-Perspective human data")
    print("  - internal_wb should show higher deontic/prosocial; public_wb should show higher competitive")
    print("  - Moral character judgment (not computed directly by this script; average the 16 character_items):")
    print("    human sample means -- no=2.70, internal=4.34, external=4.05, public=3.38 (reported in the paper)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_condition", type=int, default=20)
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--output", type=str, default="firsthand_results.csv")
    args = parser.parse_args()

    run_collection(args.n_per_condition, args.model, args.output)
    quick_sanity_check(args.output)
