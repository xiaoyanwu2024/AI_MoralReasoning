# Data and Code Release

Author: Xiaoyan Wu
Date: June 2026
Contact: xiaoyan.psych@gmail.com

This folder contains the raw LLM-generated data and the scripts used to produce them for the
study comparing human and LLM motive attributions toward whistleblowers, based on the human
benchmark data of Brotzeller, van Houwelingen, Gollwitzer, & Fischer (2025), *Personality and
Social Psychology Bulletin* (OSF: q4rsn).

Materials come in two narrative-perspective versions: a **Firsthand-Perspective** version (the
rater witnesses the events directly) and a **Secondhand-Perspective** version (the rater hears
about the events only afterward).

## Folder structure

```
code/
    firsthand.py                 shared scenario materials, scales, and DeepSeek-V3 caller
                                   (Firsthand-Perspective materials)
    firsthand_gpt4o.py            GPT-4o caller, Firsthand-Perspective
    firsthand_gemini.py           Gemini 2.5 Pro caller, Firsthand-Perspective
    firsthand_hf_llama.py         Llama-3.1-8B-Instruct caller (Hugging Face Inference API),
                                   Firsthand-Perspective
    firsthand_cerebras_llama.py   generic Cerebras-hosted-model caller (used with
                                   --model gpt-oss-120b for this study), Firsthand-Perspective

    secondhand.py                 shared scenario materials and DeepSeek-V3 caller
                                   (Secondhand-Perspective materials)
    secondhand_gpt4o.py           GPT-4o caller, Secondhand-Perspective
    secondhand_gemini.py          Gemini 2.5 Pro caller, Secondhand-Perspective
    secondhand_hf_llama.py        Llama-3.1-8B-Instruct caller, Secondhand-Perspective
    secondhand_cerebras_llama.py  generic Cerebras-hosted-model caller (used with
                                   --model gpt-oss-120b for this study), Secondhand-Perspective

data/
    firsthand_perspective/   raw CSV output for all five models, Firsthand-Perspective materials
    secondhand_perspective/  raw CSV output for all five models, Secondhand-Perspective materials
```

The `secondhand_*.py` scripts import scenario-building and scale functions from `firsthand.py`
and `secondhand.py`, so all files in `code/` should be kept in the same directory (as they are
here) for the imports to resolve.

## Installation

```
pip install -r requirements.txt
```

`requirements.txt` covers both the OpenAI-compatible client (used for DeepSeek, OpenAI, Gemini,
and Cerebras, which all expose OpenAI-compatible endpoints) and the Hugging Face client (used only
by `firsthand_hf_llama.py` / `secondhand_hf_llama.py`).

## Usage

Each script takes `--n_per_condition` and `--output` arguments, and reads its API key from an
environment variable (`DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `CEREBRAS_API_KEY`,
or `HF_TOKEN` depending on the script). Sampling temperature (0.8) and random seed (42) are fixed
as function defaults. `max_tokens` defaults are set per-script to give Gemini 2.5 Pro and
gpt-oss-120b enough headroom for their hidden reasoning pass before visible output.

The exact commands used to produce the CSV files included in `data/` were:

```
export DEEPSEEK_API_KEY=...
python firsthand.py    --model deepseek-chat --n_per_condition 200 --output firsthand_full_deepseekv3.csv
python secondhand.py   --model deepseek-chat --n_per_condition 200 --output secondhand_full_deepseekv3.csv

export OPENAI_API_KEY=...
python firsthand_gpt4o.py  --n_per_condition 200 --output firsthand_full_gpt4o.csv
python secondhand_gpt4o.py --n_per_condition 200 --output secondhand_full_gpt4o.csv

export GEMINI_API_KEY=...
python firsthand_gemini.py  --n_per_condition 200 --output firsthand_full_gemini25pro.csv
python secondhand_gemini.py --n_per_condition 200 --output secondhand_full_gemini25pro.csv

export HF_TOKEN=...
python firsthand_hf_llama.py  --n_per_condition 200 --output firsthand_full_hf_llama3.1_8b.csv
python secondhand_hf_llama.py --n_per_condition 200 --output secondhand_full_hf_llama3.1_8b.csv

export CEREBRAS_API_KEY=...
python firsthand_cerebras_llama.py  --model gpt-oss-120b --n_per_condition 200 --output firsthand_full_cerebras_gptoss120b.csv
python secondhand_cerebras_llama.py --model gpt-oss-120b --n_per_condition 200 --output secondhand_full_cerebras_gptoss120b.csv
```

Note the `--model gpt-oss-120b` override on the two `*_cerebras_llama.py` calls: these scripts
default to `llama3.1-8b` (a generic Cerebras-hosted-model caller) but were run with the override
above to collect the gpt-oss-120b data for this study.

All scripts write incrementally (one row per completed query) and support resuming an
interrupted run from the existing output CSV -- if you re-run the same command against an
existing output file, the script will only fill in the remaining gap per condition rather than
starting over (DeepSeek's `firsthand.py`/`secondhand.py` are the exception: they do not support
resuming and will overwrite the file from scratch).

## Data notes

- All ten CSV files are complete (800 rows: 200 per disclosure condition). Data collection for
  `secondhand_perspective/secondhand_full_gemini25pro.csv` was twice interrupted by Gemini API
  billing limits (a daily request quota, then prepaid credit exhaustion) before finishing; this is
  noted here only because it explains why that file's rows were collected across three separate
  runs, not because anything is incomplete.
- Column order differs slightly for the DeepSeek-V3 files (`*_deepseekv3.csv`): they were produced
  by an earlier version of the collection script that wrote columns in alphabetical order rather
  than the fixed order used by the other four models' scripts. Column *names* are identical across
  all files; only the column order differs. Downstream analysis code should select columns by name,
  not position.
- The `model` column's format is not fully consistent across files: four files tag the value with
  the access route in parentheses (e.g., `gpt-4o-2024-08-06(OpenAI)`, `gemini-2.5-pro(Gemini)`,
  `meta-llama/Llama-3.1-8B-Instruct(HF Inference API)`, `gpt-oss-120b(Cerebras)`), while the
  DeepSeek-V3 files just contain `deepseek-chat` with no parenthetical tag. This reflects script
  history rather than a difference in how the data were collected; if you need a uniform model
  label, derive it from the file name rather than relying on the `model` column's exact format.
- Each row is one simulated respondent: one LLM query, answering all scale items for one randomly
  sampled persona, one randomly sampled pair of character genders, and one of the four disclosure
  conditions.
- `secondhand_perspective/secondhand_full_hf_llama3.1_8b.csv` has one cell with a missing value
  (`competitive_3` is blank on row 610, condition `external_wb`): the model's JSON response that
  call happened to omit that field, and the collection script did not validate field completeness
  before writing the row. This is the only missing value across all 8,000 rows in this release;
  every other cell in all ten files is populated. Downstream code should handle this one blank
  cell (e.g., as `NaN` on read) rather than assume every row has all 35 fields.

## Third-party materials and license scope

The scenario text (`SCENARIO_INTRO` and `CONDITION_ENDINGS` in `firsthand.py` and `secondhand.py`)
reproduces material from the original study materials of Brotzeller, van Houwelingen, Gollwitzer,
& Fischer (2025), OSF: q4rsn, which are licensed
[**CC BY-NC-ND 4.0**](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode)
(Attribution-NonCommercial-NoDerivatives) by the original authors. That specific text is **not**
covered by this repository's MIT code license; it remains subject to the original CC BY-NC-ND 4.0
terms (noncommercial use only, no derivative materials), with attribution to the original authors.
Everything else in `code/` is original code written for this project and is MIT-licensed.

This distinction also extends, with some caution, to `data/`: the LLM-generated ratings
themselves are original synthetic outputs (numeric scale responses, not republished scenario
text), but they were produced using prompts that embed the CC BY-NC-ND-licensed scenario text. We
license the data files under CC BY 4.0 on that basis, but if you intend a commercial use of the
data, you should independently confirm this is consistent with the original materials' NC
restriction, or contact the original authors.

## Citation

If you use this code or data, please cite the associated paper (citation to be added upon
publication) and the original human benchmark study:

> Brotzeller, F., van Houwelingen, G., Gollwitzer, M., & Fischer, M. (2025). Motive attributions
> shape judgments of whistleblowers' moral characters. *Personality and Social Psychology
> Bulletin*. https://doi.org/10.1177/01461672251340111

## License

Original code (`code/`) is released under the MIT License (see `LICENSE-CODE`), with the
scenario-text exception described above. Data (`data/`) is released under CC BY 4.0 (see
`LICENSE-DATA`), subject to the same caution described above.
