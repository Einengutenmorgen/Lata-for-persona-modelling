# Codebase Overview — LATA for Persona Modelling

## What this project does

This project implements and evaluates **LATA (Layer-Adaptive Trajectory Alignment)** for personality modelling in language models. The core idea: given a LoRA adapter trained to express a personality trait, and two reference models (base + instruct), compute per-layer cosine similarity between the LoRA update direction and the instruction-tuning direction. Use this similarity to weight how much each layer is modified when injecting the personality adapter into the model — layers that are already instruction-like get less change; layers specific to the personality trait get more.

The evaluation pipeline then measures whether the resulting model actually expresses the target personality via multiple instruments: ROUGE scoring against reference responses, the IPIP-50 Big Five questionnaire, and open-ended text generation scored by a personality classifier.

---

## Pipeline Map

```
Raw CSVs
   │
   ▼
step_0_1_prepdata.py        Convert train/dev/test CSVs to JSONL
   │
   ▼
[LoRA adapter trained externally or via step_0_2_sft_seq.py]
   │
   ├──────────────────────────────────────────────────────────────┐
   ▼                                                              ▼
step_2.py                                                  step_1.py  (diagnostic only)
Compute per-layer cosine(τ_comp, τ_instr)                  Compute ||τ_instr|| per layer
→ layer_cosine.pt
   │
   ▼
step_3_weights.py
Convert cosines to layer weights (linear / log / threshold)
→ layer_weights.json
   │
   ├─────────────────────────────────────────┐
   ▼                                         ▼
step_6_lambda_gen.py                   step_4_apply_lata.py
Lambda sweep: in-memory LATA +         Save a single merged model
ROUGE-L evaluation                     to disk at a fixed λ
→ sweep_results.csv
   │
   ├──────────────────────────────────────────────────────────────────┐
   ▼                                                                  ▼
step_5_eval_gen.py                                          step_9_big5_inventory.py
Single-model ROUGE-L evaluation                             Run IPIP-50 Big Five questionnaire
→ preds.csv                                                 → {out}.responses.csv, {out}.details.csv
                                                               │
                                                               ▼
                                                         step_10_eval_questionaires.py
                                                         Score IPIP-50 responses
                                                         → seq_profiles.csv

step_11_OEG_personality_text_gen_simple.py
Open-ended generation + KevSun/Personality_LM scoring
→ profiles_{ts}.csv, generations_{ts}.csv
```

---

## File-by-File Reference

### step_0_1_prepdata.py — Data Preparation
**Status: Active**

Converts pre-split CSVs into JSONL files consumed by the rest of the pipeline.

**Inputs:** `{data_root}/{task}/{task}_{train,dev,test}.csv`  
Each CSV must have columns: `prompt`, `answer` (A or B), `trait`, `level`

**Outputs:** `{out_root}/{task}/{train,dev,test}.jsonl`  
Each JSONL record contains: `prompt`, `chosen`, `rejected`, `trait`, `level`, `story`, `source_answer`, `source_row`

**Key logic:**
- Parses `Story: ... Options: A. ... B. ...` from the raw prompt column via regex
- Rebuilds prompt as: `Trait: {trait} / Level: {level} / Story: {story} / Write a single reply...`
- Sets `chosen` = the correct option text, `rejected` = the incorrect option text

**Functions:** `parse_story_and_options()`, `build_structured_prompt()`, `read_and_convert()`, `write_jsonl()`

---

### step_0_2_sft_seq.py — LoRA SFT Training
**Status: Uncertain — may be obsolete**

Trains LoRA adapters on personality response data using TRL's `SFTTrainer`.

**Inputs:** `Task_III/{task}/train.jsonl`, `dev.jsonl` (output of step_0_1)  
**Outputs:** LoRA adapters saved to `models_llama31_8b_sft/{task}/`

**Config (all hardcoded):**
- Base model: `meta-llama/Llama-3.1-8B-Instruct`
- CUDA device: GPU 1
- LoRA: r=8, alpha=16, target modules: q/k/v/o/gate/up/down_proj
- Tasks list: only `agreeableness_high`, `agreeableness_low` (incomplete)

**Issues:**
- Everything hardcoded — no argparse
- Contains commented-out old `to_prompt_completion()` approach
- The TASKS list has only 2 entries; clearly a partial/experimental run
- The user notes that LoRA fine-tuning was "removed later" — verify whether this script is still needed

---

### step_1.py — τ_instr Norms (Diagnostic)
**Status: Diagnostic only — not required by downstream steps**

Computes the per-layer L2 norm of `τ_instr = instruct_weights − base_weights`.

**Inputs:** `--base` (base model), `--instruct` (instruct model)  
**Outputs:** `artifacts/instr_stats.pt` — dict with `per_layer_norm`, `total_norm`

**Note:** step_2.py recomputes these values inline during the cosine similarity calculation. step_1 is useful for standalone analysis of how much instruction tuning changed each layer, but its output file is not consumed by any other script.

**Duplicated code:** `TARGET_SUBSTRINGS`, `LAYER_RE`, `layer_id()`, `is_target_param()` — identical in step_1.2, step_2, step_4, step_6_lambda, step_6_lambda_gen.

---

### step_1.2.py — τ_comp Norms (Diagnostic)
**Status: Diagnostic only — not required by downstream steps**

Computes the per-layer Frobenius norm of `τ_comp = B @ A * (alpha/r)` (the dense LoRA delta).

**Inputs:** `--adapter_dir` (LoRA adapter folder)  
**Outputs:** `artifacts/tau_comp_stats.pt` — dict with `per_layer_norm`, `total_norm`

**Note:** step_2.py also recomputes this inline. This script is useful for comparing LoRA update magnitudes across layers in isolation, but its output is not consumed by any other script.

**Duplicated code:** Same as step_1 plus `get_lora_scaling()`.

---

### step_2.py — Layer Cosine Similarity (Core LATA Step)
**Status: Active — core pipeline step**

Computes per-layer cosine similarity between τ_comp (LoRA direction) and τ_instr (instruction tuning direction). This is the central computation of LATA.

**Inputs:** `--adapter_dir`, `--base` (base model), `--instruct` (instruct model)  
**Outputs:** `artifacts/layer_cosine.pt` — dict with `layer_cosine` (int → float)

**Key logic:**
- For each matched lora_A/lora_B pair on target modules:
  - τ_comp = B @ A * scaling
  - τ_instr = instruct_weight − base_weight
  - Accumulates dot product and norm² per layer (fp32 for stability)
- Computes cosine = dot / (||τ_comp|| × ||τ_instr||) per layer

**Duplicated code:** `TARGET_SUBSTRINGS`, `LAYER_RE`, `layer_id()`, `is_target_param()`, `get_lora_scaling()`, `clean_lora_key()`.

---

### step_3_weights.py — Layer Weight Calculation
**Status: Active**

Converts per-layer cosine similarities into scalar weights for LATA application.

**Inputs:** `--in_pt` (layer_cosine.pt), `--method` (linear / log / threshold), `--sigma`  
**Outputs:** `--out_json` — JSON with `layer_weight` dict (int → float)

**Weight schemes:**
- `linear`: rank/L — layers with highest cosine get weight 1/L (least modified), lowest cosine get weight 1.0 (most modified)
- `log`: log_L(rank) — similar ranking, logarithmic scale
- `threshold`: 0.0 if cosine ≥ sigma, else 1.0 (binary mask)

**Interpretation:** High cosine = layer already encodes instruction-like direction → apply less personality change. Low cosine = layer is personality-specific → apply more change.

---

### step_4_apply_lata.py — Apply LATA (Save Merged Model)
**Status: Active — use when you need a saved merged model**

Applies LATA to produce a saved HuggingFace model on disk.

**Inputs:** `--adapter_dir`, `--target_model` (usually instruct), `--weights_json`, `--lambda_`  
**Outputs:** Full merged model saved to `--out_dir`

**Key logic:**
- For each LoRA pair: `weight += lambda_ × w[layer] × (B @ A × scaling)`
- Saves model + tokenizer in HF format

**Use case:** When you need a persistent merged model for deployment or for use with step_5_eval_gen or step_9. For lambda sweeps, step_6_lambda_gen is more efficient (in-memory, no disk I/O).

**Duplicated code:** Same 6 functions/constants as step_2.

---

### step_5_eval.py — Classification Evaluation
**Status: BROKEN — do not use**

Intended to evaluate models on A/B multiple-choice personality questions.

**Bugs:**
1. `build_prompt()` accesses `example["messages"]` — field does not exist in the current JSONL format (leftover from an old TASK_II format)
2. `return f"{pompt}"` — `pompt` is undefined (typo); would raise `NameError` at runtime
3. `ex.get("correct_option", None)` — field does not exist in current JSONL (it's `source_answer`)
4. Hardcoded `os.environ["CUDA_VISIBLE_DEVICES"] = "1"` at module level

This script reflects an abandoned classification-based evaluation approach (Task II format). It has never worked with the current data format.

---

### step_5_eval_gen.py — Generation Evaluation (Single Model)
**Status: Active**

Evaluates a single model on generation quality using ROUGE-L against reference responses.

**Inputs:** `--model_dir`, `--test_jsonl` (uses `prompt` and `chosen` fields)  
**Outputs:** `artifacts/gen_preds.csv` — per-example `generated`, `rouge_l`

**Key logic:** Direct completion (no chat template) — generates from `prompt`, scores against `chosen` with ROUGE-L.

---

### step_6_lambda.py — Classification Lambda Sweep
**Status: BROKEN — do not use**

Intended to sweep λ values and evaluate via A/B classification.

**Bugs:**
1. `gold = ex.get('chosen', None)` — `chosen` contains free text, not "A"/"B"; the check `if gold in ("A", "B")` will always fail, making `total` always 0
2. `print(f'Prompt: {prompt} \n Gold: {gold}')` inside the eval loop — would print every single example
3. Hardcoded `os.environ["CUDA_VISIBLE_DEVICES"] = "1"` at module level

This script reflects the abandoned classification evaluation track. The generation-based approach (step_6_lambda_gen) replaced it.

---

### step_6_lambda_gen.py — Generation Lambda Sweep
**Status: Active — main sweep script**

Sweeps multiple λ values and evaluates LATA-merged models using ROUGE-L.

**Inputs:** `--adapter_dir`, `--weights_json`, `--test_jsonl`, `--base_model`, `--instruct_model`, `--lambdas`  
**Outputs:** `artifacts/sweep_gen.csv` — per-run `rouge_l` at each λ

**Key logic:**
- Evaluates base and instruct models as baselines
- Precomputes `w[layer] × (B @ A × scaling)` for all LoRA pairs and caches on GPU
- For each λ: applies `param = orig + λ × cached_delta`, evaluates, restores original weights
- Efficient: avoids reloading the model for each λ value

**Duplicated code:** Same LoRA utility functions as step_2 and step_4.

---

### step_8_combine.py — Combine Sweep Results
**Status: Dead — hardcoded paths, no argparse**

Concatenates multiple lambda sweep CSVs into one DataFrame.

**Hardcoded paths:**
```
artifacts/agreeableness_high_linear_lambda_sweep.csv
artifacts/agreeableness_high_log_lambda_sweep.csv
artifacts/agreeableness_high_thr2e-4_lambda_sweep.csv
```

This is essentially a 17-line notebook cell. The paths no longer exist. If needed, replace with a simple CLI tool or notebook cell.

---

### step_9_big5_inventory.py — IPIP-50 Big Five Questionnaire
**Status: Active**

Runs the 50-item IPIP Big Five inventory across multiple models and collects Likert responses.

**Inputs:** `--questionnaire` (JSON file), `--out` (output path prefix)  
**Outputs:** `{out}.details.csv` (per-question), `{out}.responses.csv` (wide format, X_1..X_50)

**Key logic:**
- Uses raw text completion format (not chat template) for both base and instruct models
- Prompt: scale definition + statement + "Answer:" (expects a single digit 1-5)
- Parses response with regex for digits 1-5, falls back to word matching
- Repeats each run `--repeats` times with different seeds for stability

**Issues:**
- Model list is hardcoded inside `main()` with many commented-out entries — should be CLI arg or config file
- `os.environ["CUDA_VISIBLE_DEVICES"] = "3"` hardcoded at module level
- `infer_llama_config_if_missing()` / `try_load_as_peft_adapter()` are workarounds for specific local model issues; may not be needed with clean HF models

---

### step_10_eval_questionaires.py — Score IPIP-50 Responses
**Status: Active but fragile**

Applies IPIP-50 scoring keys to convert raw Likert responses into Big Five trait scores.

**Inputs:** Hardcoded `seq.responses.csv` (output of step_9)  
**Outputs:** Hardcoded `seq_profiles.csv`

**Key logic:**
- Scoring keys per trait (10 items each, some reverse-scored)
- Reverse scoring: `6 - score` for negatively-keyed items
- Aggregates mean score per trait per model

**Issues:**
- Hardcoded input/output paths — should accept CLI args
- No argparse whatsoever

---

### step_11_OEG_personality_text_gen_simple.py — Open-Ended Generation Evaluation
**Status: Active — most complete script**

Evaluates personality expression through open-ended text generation, scored by a Big Five personality classifier.

**Inputs:** `--model_path` or `--config` (JSON list of models)  
**Outputs:** `profiles_{timestamp}.csv`, `generations_{timestamp}.csv`

**Key logic:**
- 12 personality-relevant sentence-completion prompts (e.g. "I like to", "At parties, I")
- Two prompt formats: chat-template (instruct/SFT models) vs. raw completion (base models)
- Uses `KevSun/Personality_LM` (sequence classification) to score each generated text on Big Five
- Computes mean personality profile across all generated texts per model
- Supports batch evaluation via JSON config file

**Classes:** `PersonalityLM` (wrapper around KevSun/Personality_LM), `TextGenerator` (generation with seed control)

---

## Duplicated Code — Candidates for utils.py

The following are defined identically in **5–6 files** (step_1, step_1.2, step_2, step_4, step_6_lambda, step_6_lambda_gen):

| Symbol | Appears in |
|--------|-----------|
| `TARGET_SUBSTRINGS` tuple | step_1, step_1.2, step_2, step_4, step_6_lambda, step_6_lambda_gen |
| `LAYER_RE` regex | step_1, step_1.2, step_2, step_4, step_6_lambda, step_6_lambda_gen |
| `layer_id(name)` | step_1, step_1.2, step_2, step_4, step_6_lambda, step_6_lambda_gen |
| `is_target_param(name)` | step_1, step_1.2, step_2, step_4, step_6_lambda, step_6_lambda_gen |
| `get_lora_scaling(adapter_dir)` | step_1.2, step_2, step_4, step_6_lambda, step_6_lambda_gen |
| `clean_lora_key(kA)` | step_2, step_4, step_6_lambda, step_6_lambda_gen |
| `load_tokenizer(path)` | step_6_lambda, step_6_lambda_gen |

All seven belong in a single `utils.py` module.

---

## Dead / Broken Code Summary

| File | Status | Reason |
|------|--------|--------|
| `step_5_eval.py` | **Broken** | Undefined variable `pompt`; wrong field names (`messages`, `correct_option`); old TASK_II format |
| `step_6_lambda.py` | **Broken** | `chosen` field contains text, not "A"/"B"; classification track abandoned |
| `step_8_combine.py` | **Dead** | Hardcoded artifact paths that no longer exist |
| `step_0_2_sft_seq.py` | **Uncertain** | User reports LoRA training was removed; hardcoded, incomplete TASKS list |

## Diagnostic-Only Scripts (not part of main pipeline)

| File | Purpose | Output used by |
|------|---------|----------------|
| `step_1.py` | Compute `||τ_instr||` per layer | Nothing — step_2 recomputes inline |
| `step_1.2.py` | Compute `||τ_comp||` per layer | Nothing — step_2 recomputes inline |

## Hardcoded Configs to Externalize

| File | Hardcoded value |
|------|----------------|
| `step_0_2_sft_seq.py` | `CUDA_VISIBLE_DEVICES=1`, model path, TASKS list, all hyperparams |
| `step_5_eval.py` | `CUDA_VISIBLE_DEVICES=1` |
| `step_6_lambda.py` | `CUDA_VISIBLE_DEVICES=1` |
| `step_9_big5_inventory.py` | `CUDA_VISIBLE_DEVICES=3`, model list |
| `step_10_eval_questionaires.py` | input/output file paths |
| `step_8_combine.py` | input file paths |

---

## Recommended Refactoring Steps

1. **Create `utils.py`** — extract the 7 duplicated symbols listed above
2. **Delete `step_5_eval.py`** and **`step_6_lambda.py`** — both broken; the generation-based equivalents replace them
3. **Delete or replace `step_8_combine.py`** — hardcoded dead code; replace with a 3-line notebook cell or CLI script if needed
4. **Decide on `step_0_2_sft_seq.py`** — if LoRA training was removed from the approach, delete it; if still needed, add argparse and move config to CLI args
5. **Decide on `step_1.py` and `step_1.2.py`** — if diagnostic analysis is still useful, keep; otherwise delete (their core logic is already in step_2)
6. **Add argparse to `step_10_eval_questionaires.py`** — replace hardcoded paths
7. **Move model list in `step_9`** to a CLI arg or JSON config file
8. **Remove all top-level `os.environ["CUDA_VISIBLE_DEVICES"]`** — replace with CLI `--device` args or document as env var to set before running
