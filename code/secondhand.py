# ---------------------------------------------------------------------------
# Author: Xiaoyan Wu
# Date: June 2026
# Contact: xiaoyan.psych@gmail.com
# ---------------------------------------------------------------------------

"""
Data-collection script: LLM simulated respondents vs. Secondhand-Perspective human data
(the "heard about it afterward" framing).

Key differences from firsthand.py:
- Different scenario framing: the respondent "happened to hear about" what the team did some
  time ago, rather than witnessing it directly.
- Personas are sampled to match the real demographic composition of the Secondhand-Perspective
  human sample: age ~ N(49.43, 15.06), gender 64.3% female / 34.8% male / 0.9% other, 5.66%
  students.
- Reuses the scale items, system-prompt construction, and call_model (DeepSeek) logic from
  firsthand.py.

Usage:
    export DEEPSEEK_API_KEY=your_key
    python secondhand.py --n_per_condition 200
"""

import argparse
import csv
import json
import random
import re
import time

from firsthand import (
    build_user_prompt,
    call_model,
    CHARACTER_ITEMS,
)

# ---------------------------------------------------------------------------
# Secondhand-Perspective scenario materials (adapted from the OSF "Study Material" PDF for the
# secondhand-perspective version). Key difference from Firsthand-Perspective: the opening frame
# is "happened to hear about it afterward", not "witnessed directly".
#
# LICENSE NOTE: SCENARIO_INTRO and CONDITION_ENDINGS below reproduce text from the original
# study materials of Brotzeller, van Houwelingen, Gollwitzer, & Fischer (2025), OSF: q4rsn,
# which are licensed CC BY-NC-ND 4.0 (Attribution-NonCommercial-NoDerivatives;
# https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode). This text is NOT covered by this
# repository's MIT license -- it remains governed by the original CC BY-NC-ND 4.0 terms
# (noncommercial use only, no derivative materials), with attribution to the original authors.
# All other code in this repository (everything outside this text block) is MIT-licensed.
# ---------------------------------------------------------------------------

SCENARIO_INTRO = """You are a physician at the hospital "Klinik am See". You have already been working \
with your team in this hospital for several years. Together with your team, you advise people who are \
ill on a daily basis, make diagnoses and carry out treatments.

The other day you happened to hear about something that happened on your team some time ago:

{perpetrator_ref} from your team, Dr Schmitt, billed the health insurance company for more services than \
were actually provided for some patients. For example, services were billed by Dr Schmitt without \
{perpetrator_pronoun} even being present during the billed treatments."""

CONDITION_ENDINGS = {
    "no_wb": (
        "You also learned that {wb_ref} of your colleagues, Dr Bauer, learned about it and "
        "subsequently did not pass on any information about it."
    ),
    "internal_wb": (
        "You also learned that {wb_ref} of your colleagues, Dr Bauer, learned about it and "
        "subsequently passed on information about it to the hospital's compliance department, which is "
        "responsible for ensuring that the hospital complies with all legal requirements."
    ),
    "external_wb": (
        "You also learned that {wb_ref} of your colleagues, Dr Bauer, learned about it and "
        "subsequently passed on information about it to the relevant authorities."
    ),
    "public_wb": (
        "You also learned that {wb_ref} of your colleagues, Dr Bauer, learned about it and "
        "subsequently passed on information about it to a newspaper that is distributed throughout Germany."
    ),
}

GENDER_PRONOUNS = {
    "male": {"ref": "another physician", "pronoun": "him"},
    "female": {"ref": "another physician", "pronoun": "her"},
}
WB_GENDER_REF = {"male": "one", "female": "one"}  # In the Secondhand-Perspective materials, the discloser's gender does not change sentence structure; only "one" is used

# ---------------------------------------------------------------------------
# Persona sampling: matched to the real demographic composition of the Secondhand-Perspective
# human sample (see the paper's Method section)
# ---------------------------------------------------------------------------

def sample_persona(rng: random.Random) -> dict:
    age = round(rng.gauss(49.43, 15.06))
    age = max(18, min(85, age))
    gender = rng.choices(["female", "male", "other"], weights=[64.3, 34.8, 0.9])[0]
    is_student = rng.random() < 0.0566
    student_clause = "You are currently a university student. " if is_student else "You are not currently a student. "
    return {"age": age, "gender": gender, "is_student": is_student, "student_clause": student_clause}


def sample_scenario_genders(rng: random.Random) -> dict:
    return {
        "schmitt_gender": rng.choice(["male", "female"]),
        "bauer_gender": rng.choice(["male", "female"]),
    }


def build_scenario_text(condition: str, genders: dict) -> str:
    schmitt = GENDER_PRONOUNS[genders["schmitt_gender"]]
    intro = SCENARIO_INTRO.format(perpetrator_ref=schmitt["ref"], perpetrator_pronoun=schmitt["pronoun"])
    ending = CONDITION_ENDINGS[condition].format(wb_ref=WB_GENDER_REF[genders["bauer_gender"]])
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


def parse_json_response(raw_text: str) -> dict:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON in the model output: {raw_text[:200]}")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# Main loop: writes the CSV incrementally (fixes the earlier issue where data was only saved
# at the very end, so an interruption lost everything)
# ---------------------------------------------------------------------------

def run_collection(n_per_condition: int, model: str, output_csv: str, seed: int = 42):
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
                    print(f"[condition={condition} attempt {i+1}] failed: {e}")
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
                time.sleep(0.3)
    finally:
        if f_out:
            f_out.close()

    print(f"\nSaved {n_done} rows to {output_csv}")


def quick_sanity_check(output_csv: str):
    import statistics

    with open(output_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    dims = ["deontic", "prosocial", "individualistic", "competitive"]
    print("\n=== Secondhand-Perspective: mean rating by condition x motive dimension (LLM simulated respondents) ===")
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
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--output", type=str, default="secondhand_full_deepseekv3.csv")
    args = parser.parse_args()

    run_collection(args.n_per_condition, args.model, args.output)
    quick_sanity_check(args.output)
