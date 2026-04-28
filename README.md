# Codebase Overview — LATA for Persona Modelling

## What this project does

This project implements and evaluates **LATA (Layer-Adaptive Trajectory Alignment)** for personality modelling in language models. The core idea: given a full-parameter fine-tuned model that expresses a personality trait, and two reference models (base + instruct), compute per-layer cosine similarity between the personality tuning direction (τ_comp = finetuned − base) and the instruction tuning direction (τ_instr = instruct − base). Use this similarity to weight how much each layer is modified when injecting the personality update — layers that are already instruction-like get less change; layers specific to the personality trait get more.

The evaluation pipeline measures whether the resulting model expresses the target personality via three instruments: ROUGE-L scoring against reference responses, the IPIP-50 Big Five questionnaire, and open-ended text generation scored by a personality classifier.

> **Note on naming:** The codebase contains many references to "LoRA" (variable names, argument names, comments). The project originally used LoRA fine-tuning but was switched to full-parameter fine-tuning. The LoRA names are cosmetic leftovers in active code. Three scripts (`step_0_2`, `step_1.2`, the LoRA-specific loading logic in `step_2`/`step_4`/`step_6_lambda_gen`) contain **functional** LoRA dependencies that are now broken or obsolete — see per-file notes below.

---

## Pipeline Map

```
Raw CSVs
   │
   ▼
step_0_1_prepdata.py        Convert train/dev/test CSVs to JSONL
   │
   ▼
[Full-parameter fine-tuned personality model trained externally]
   │
   ├──────────────────────────────────────────────────────────────┐
   ▼                                                              ▼
step_2.py                                                  step_1.py  (diagnostic only)
Compute per-layer cosine(τ_comp, τ_instr)                  Compute ||τ_instr|| per layer
  τ_comp = finetuned − base  ← NEEDS UPDATE                → instr_stats.pt
  (currently reads lora_A/lora_B — broken for full-param)
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
  ← NEEDS UPDATE (lora_A/lora_B)         ← NEEDS UPDATE (lora_A/lora_B)
→ sweep_results.csv
   │
   ├──────────────────────────────────────────────────────────────────┐
   ▼                                                                  ▼
step_5_eval_gen.py                                          step_9_big5_inventory.py
Single-model ROUGE-L evaluation                             Run IPIP-50 Big Five questionnaire
→ gen_preds.csv                                             → {out}.responses.csv, {out}.details.csv
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
Each JSONL record: `prompt`, `chosen`, `rejected`, `trait`, `level`, `story`, `source_answer`, `source_row`

**Key logic:**
- Parses `Story: ... Options: A. ... B. ...` from the raw prompt column via regex
- Rebuilds prompt as: `Trait: {trait} / Level: {level} / Story: {story} / Write a single reply...`
- `chosen` = correct option text, `rejected` = incorrect option text

**Functions:** `parse_story_and_options()`, `build_structured_prompt()`, `read_and_convert()`, `write_jsonl()`

---

### step_0_2_sft_seq.py — LoRA SFT Training
**Status: OBSOLETE — delete**

Was used to train LoRA adapters. The project has moved to full-parameter fine-tuning done externally. This script uses `LoraConfig` + `peft` and produces LoRA adapters, which the rest of the pipeline no longer consumes.

Additional problems even if LoRA were still in use:
- Everything hardcoded (no argparse) — CUDA device, model path, hyperparameters
- TASKS list has only 2 entries; clearly a partial/experimental run
- Contains commented-out old `to_prompt_completion()` format

---

### step_1.py — τ_instr Norms (Diagnostic)
**Status: Diagnostic — not required by downstream steps, but still valid**

Computes the per-layer L2 norm of `τ_instr = instruct_weights − base_weights`.

**Inputs:** `--base`, `--instruct`  
**Outputs:** `artifacts/instr_stats.pt` — `per_layer_norm`, `total_norm`

step_2 recomputes τ_instr inline, so this output is not consumed by anything. Useful for standalone inspection of how much instruction tuning shifted each layer.

**Contains duplicated code:** `TARGET_SUBSTRINGS`, `LAYER_RE`, `layer_id()`, `is_target_param()` — shared with step_2, step_4, step_6_lambda_gen → move to `utils.py`.

---

### step_1.2.py — τ_comp Norms via LoRA
**Status: OBSOLETE — delete**

Computes the per-layer Frobenius norm of `τ_comp = B @ A * (alpha/r)` — the dense LoRA delta. With full-parameter tuning, there are no `lora_A`/`lora_B` matrices. The equivalent computation is now `τ_comp = finetuned_weight − base_weight`, which is structurally identical to what step_1 does for τ_instr. If τ_comp norms are needed for diagnostics, step_1 can be called with `--base <base_model> --instruct <finetuned_model>`.

---

### step_2.py — Layer Cosine Similarity (Core LATA Step)
**Status: Active, but needs update for full-parameter tuning**

Computes per-layer cosine similarity between τ_comp (personality tuning direction) and τ_instr (instruction tuning direction). This is the central computation of LATA.

**Inputs:** `--adapter_dir`, `--base`, `--instruct`  
**Outputs:** `artifacts/layer_cosine.pt` — `layer_cosine` (int → float)

**Current behaviour (broken for full-param):**
- Opens `adapter_model.safetensors`, loops over `lora_A` keys
- Computes τ_comp as `B @ A * (alpha/r)` — this requires LoRA matrices that do not exist in a full-param fine-tuned model

**What it should do instead:**
- Accept `--finetuned` (path to full-param fine-tuned model) instead of `--adapter_dir`
- Compute τ_comp = `finetuned_weight − base_weight` for each target parameter (same arithmetic as τ_instr)
- The cosine accumulation logic (dot product, norm², per-layer) is correct and unchanged

**Functions to remove after update:** `get_lora_scaling()`, `clean_lora_key()` — LoRA-specific, no longer needed  
**Functions to keep:** `layer_id()`, `is_target_param()` — still needed → move to `utils.py`

---

### step_3_weights.py — Layer Weight Calculation
**Status: Active — no changes needed**

Converts per-layer cosine similarities into scalar weights for LATA application.

**Inputs:** `--in_pt` (layer_cosine.pt from step_2), `--method` (linear/log/threshold), `--sigma`  
**Outputs:** `--out_json` — JSON with `layer_weight` (int → float)

**Weight schemes:**
- `linear`: rank/L — highest cosine → smallest weight (least modified)
- `log`: log_L(rank) — same ranking, logarithmic scale
- `threshold`: 0.0 if cosine ≥ sigma, else 1.0 (binary mask)

**Interpretation:** High cosine = layer already encodes instruction-like direction → apply less personality change. Low cosine = layer is personality-specific → apply more change.

---

### step_4_apply_lata.py — Apply LATA (Save Merged Model)
**Status: Active, but needs update for full-parameter tuning**

Applies LATA and saves the merged model to disk.

**Inputs:** `--adapter_dir`, `--target_model` (instruct), `--weights_json`, `--lambda_`  
**Outputs:** Full merged model in HF format at `--out_dir`

**Current behaviour (broken for full-param):**
- Reads `adapter_model.safetensors` for `lora_A`/`lora_B` pairs
- Applies: `weight += lambda_ × w[layer] × (B @ A × scaling)`

**What it should do instead:**
- Accept `--finetuned` instead of `--adapter_dir`
- Load finetuned model alongside target model
- Apply: `weight += lambda_ × w[layer] × (finetuned_weight − base_weight)`

**Use case:** Produces a persistent saved model. For lambda sweeps, step_6_lambda_gen is more efficient (in-memory, no disk I/O).

**Functions to remove after update:** `get_lora_scaling()`, `clean_lora_key()`  
**Functions to keep/move to utils.py:** `layer_id()`, `is_target_param()`

---

### step_5_eval.py — Classification Evaluation
**Status: BROKEN — delete**

Written for an old TASK_II data format. Has never worked with the current data format.

**Bugs:**
1. `build_prompt()` reads `example["messages"]` — field does not exist in current JSONL
2. `return f"{pompt}"` — `pompt` undefined (typo), raises `NameError` at runtime
3. `ex.get("correct_option", None)` — field does not exist (current field is `source_answer`)
4. Hardcoded `os.environ["CUDA_VISIBLE_DEVICES"] = "1"` at module level

The generation-based evaluation (step_5_eval_gen) replaces this.

---

### step_5_eval_gen.py — Generation Evaluation (Single Model)
**Status: Active**

Evaluates a single model on generation quality using ROUGE-L against reference responses.

**Inputs:** `--model_dir`, `--test_jsonl` (uses `prompt` and `chosen` fields)  
**Outputs:** `artifacts/gen_preds.csv` — per-example `generated`, `rouge_l`

Direct completion (no chat template) — generates from `prompt`, scores against `chosen`.

---

### step_6_lambda.py — Classification Lambda Sweep
**Status: BROKEN — delete**

Written for classification (A/B) evaluation. Broken and superseded by step_6_lambda_gen.

**Bugs:**
1. `gold = ex.get('chosen', None)` — `chosen` contains free text, not "A"/"B"; `if gold in ("A", "B")` always fails, `total` stays 0
2. `print(f'Prompt: {prompt} \n Gold: {gold}')` inside the eval loop floods stdout
3. Contains the same LoRA-specific loading code as step_2/step_4 (additionally broken for full-param)

---

### step_6_lambda_gen.py — Generation Lambda Sweep
**Status: Active, but needs update for full-parameter tuning**

Sweeps multiple λ values and evaluates LATA-merged models in-memory using ROUGE-L. The most efficient sweep script — precomputes and caches all deltas, restores weights between lambda values without reloading the model.

**Inputs:** `--adapter_dir`, `--weights_json`, `--test_jsonl`, `--base_model`, `--instruct_model`, `--lambdas`  
**Outputs:** `artifacts/sweep_gen.csv` — `rouge_l` per run per λ

**Current behaviour (broken for full-param):**
- Reads `adapter_model.safetensors` for `lora_A`/`lora_B` pairs
- Computes cached delta as `w[layer] × (B @ A × scaling)`

**What it should do instead:**
- Accept `--finetuned` instead of `--adapter_dir`
- Cache delta as `w[layer] × (finetuned_weight − base_weight)` per parameter

The evaluation loop (`eval_model()`), ROUGE scoring, and cache-restore sweep logic are all correct and unchanged.

**Functions to remove after update:** `get_lora_scaling()`, `clean_lora_key()`  
**Functions to keep/move to utils.py:** `layer_id()`, `is_target_param()`, `load_tokenizer()`

---

### step_8_combine.py — Combine Sweep Results
**Status: Dead — delete**

Concatenates lambda sweep CSVs into one DataFrame. The paths are hardcoded to files that no longer exist. There is no argparse. Replace with a notebook cell or a 5-line CLI script if needed.

---

### step_9_big5_inventory.py — IPIP-50 Big Five Questionnaire
**Status: Active**

Runs the 50-item IPIP Big Five inventory across multiple models and collects Likert responses.

**Inputs:** `--questionnaire` (JSON), `--out` (output path prefix)  
**Outputs:** `{out}.details.csv` (per-question), `{out}.responses.csv` (wide format X_1..X_50)

**Key logic:**
- Raw text completion format — works for both base and full-param fine-tuned models
- Prompt: scale definition + statement + "Answer:" (expects a single digit 1–5)
- Regex parses 1–5 response; falls back to word matching ("strongly agree" etc.)
- `--repeats` runs per model with different seeds for stability

**Issues:**
- Model list hardcoded inside `main()` with many commented-out entries → should be `--models` CLI arg or JSON config
- `os.environ["CUDA_VISIBLE_DEVICES"] = "3"` hardcoded at module level
- `try_load_as_peft_adapter()` attempts to load PEFT adapters — dead path for full-param models; can be removed

---

### step_10_eval_questionaires.py — Score IPIP-50 Responses
**Status: Active but fragile**

Applies IPIP-50 scoring keys to compute Big Five trait scores from step_9 output.

**Inputs:** Hardcoded `seq.responses.csv`  
**Outputs:** Hardcoded `seq_profiles.csv`

**Key logic:** 10-item scoring keys per trait; reverse-scores negatively-keyed items (`6 − score`); aggregates mean per model.

**Issues:**
- Input and output paths hardcoded — no argparse
- Entirely dependent on step_9 using the filename `seq.responses.csv`

---

### step_11_OEG_personality_text_gen_simple.py — Open-Ended Generation Evaluation
**Status: Active — most complete script**

Evaluates personality expression through open-ended text generation scored by a Big Five classifier.

**Inputs:** `--model_path` (single model) or `--config` (JSON list of models)  
**Outputs:** `profiles_{timestamp}.csv`, `generations_{timestamp}.csv`

**Key logic:**
- 12 personality-relevant sentence-completion prompts ("I like to", "At parties, I", etc.)
- Chat-template format for instruct/SFT models; raw completion for base models
- Scores each generation with `KevSun/Personality_LM` (sequence classifier → Big Five probs)
- Computes mean personality profile across all generated texts per model

**Classes:** `PersonalityLM`, `TextGenerator`

---

## Duplicated Code — Candidates for utils.py

These symbols are currently copy-pasted across active scripts. After removing the broken/obsolete files and updating step_2/step_4/step_6_lambda_gen, the following belong in a single `utils.py`:

| Symbol | Currently in | Notes |
|--------|-------------|-------|
| `TARGET_SUBSTRINGS` | step_1, step_2, step_4, step_6_lambda_gen | Keep — still needed |
| `LAYER_RE` | step_1, step_2, step_4, step_6_lambda_gen | Keep — still needed |
| `layer_id(name)` | step_1, step_2, step_4, step_6_lambda_gen | Keep — still needed |
| `is_target_param(name)` | step_1, step_2, step_4, step_6_lambda_gen | Keep — still needed |
| `load_tokenizer(path)` | step_6_lambda, step_6_lambda_gen | Keep (step_6_lambda_gen only after cleanup) |
| `get_lora_scaling(adapter_dir)` | step_1.2, step_2, step_4, step_6_lambda, step_6_lambda_gen | **Delete** — LoRA-specific, not needed after full-param update |
| `clean_lora_key(kA)` | step_2, step_4, step_6_lambda, step_6_lambda_gen | **Delete** — LoRA-specific, not needed after full-param update |

---

## Summary: What to Delete, Update, and Keep

### Delete (confirmed obsolete or broken)
| File | Reason |
|------|--------|
| `step_0_2_sft_seq.py` | LoRA SFT training — approach removed |
| `step_1.2.py` | LoRA-specific τ_comp norms — approach removed |
| `step_5_eval.py` | Broken (undefined variable, wrong fields) — classification track abandoned |
| `step_6_lambda.py` | Broken (classification track abandoned) + LoRA-specific |
| `step_8_combine.py` | Hardcoded dead code |

### Update (functionally broken for full-param tuning)
| File | Required change |
|------|----------------|
| `step_2.py` | Replace `--adapter_dir` + `lora_A/lora_B` loading with `--finetuned` + direct weight diff |
| `step_4_apply_lata.py` | Same: replace LoRA delta with `finetuned_weight − base_weight` |
| `step_6_lambda_gen.py` | Same: replace LoRA delta cache with `finetuned_weight − base_weight` |

### Minor fixes (cosmetic / config issues)
| File | Fix needed |
|------|-----------|
| `step_9_big5_inventory.py` | Move model list to CLI arg; remove `try_load_as_peft_adapter()`; remove hardcoded CUDA device |
| `step_10_eval_questionaires.py` | Add argparse for input/output paths |

### Active and correct — no changes needed
- `step_0_1_prepdata.py`
- `step_1.py`
- `step_3_weights.py`
- `step_5_eval_gen.py`
- `step_11_OEG_personality_text_gen_simple.py`

---

## Recommended Implementation Order

1. **Delete** the 5 confirmed-dead files
2. **Create `utils.py`** with the 5 surviving shared symbols
3. **Update step_2** — replace LoRA loading with full-param diff; import from utils
4. **Update step_4** — same
5. **Update step_6_lambda_gen** — same
6. **Fix step_9** — CLI model list, remove PEFT path
7. **Fix step_10** — add argparse
