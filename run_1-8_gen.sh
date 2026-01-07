#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIG (edit as needed)
# =========================

# GPU Setup
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# Verify GPU visibility
echo "------------------------------------------------"
echo "DEBUG: CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"
echo "------------------------------------------------"

# Models
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.2-1B}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"

# Paths (relative to repo root)
MODELS_DIR="${MODELS_DIR:-models_personality_sft}"          # contains per-task LoRA adapters
TASKS_ROOT="${TASKS_ROOT:-Task_III}"                        # contains per-task jsonl splits
ART_DIR="${ART_DIR:-artifacts}"
MERGED_DIR="${MERGED_DIR:-models_seq}"

# Runtime options
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
# Increased token limit for generation (sentences), not just A/B
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}" 
LAMBDAS="${LAMBDAS:-0.0,0.25,0.5,1.0,2.0}"

# Strategies to run (linear, log, threshold)
RUN_LINEAR="${RUN_LINEAR:-1}"
RUN_LOG="${RUN_LOG:-1}"
RUN_THRESHOLD="${RUN_THRESHOLD:-1}"

# Threshold sigma
SIGMA="${SIGMA:-0.0002}"

# =========================
# Helpers
# =========================

die() { echo "ERROR: $*" >&2; exit 1; }

need_file() { [[ -f "$1" ]] || die "Missing file: $1"; }
need_dir()  { [[ -d "$1" ]] || die "Missing dir: $1"; }

run_py() {
  echo
  echo ">>> $*"
  CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python "$@"
}

# =========================
# Preconditions
# =========================
# Ensure we have the GEN (generation) versions of the evaluation scripts
need_file "step_1.py"
need_file "step_1.2.py"
need_file "step_2.py"
need_file "step_3_weights.py"
need_file "step_4_apply_lata.py"
need_file "step_5_eval_gen.py"   # UPDATED
need_file "step_6_lambda_gen.py" # UPDATED

need_dir "$MODELS_DIR"
need_dir "$TASKS_ROOT"

mkdir -p "$ART_DIR" "$MERGED_DIR"

# =========================
# STEP 1 (global): instr stats
# =========================
echo "=== STEP 1: instruction vector stats (global) ==="
run_py step_1.py \
  --base "$BASE_MODEL" \
  --instruct "$INSTRUCT_MODEL" \
  --out "$ART_DIR/instr_stats.pt"

# =========================
# Iterate tasks
# =========================
TASK_LIST=()
while IFS= read -r -d '' d; do
  TASK_LIST+=("$(basename "$d")")
done < <(find "$TASKS_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

[[ "${#TASK_LIST[@]}" -gt 0 ]] || die "No task directories found under: $TASKS_ROOT"

echo
echo "Found tasks:"
printf ' - %s\n' "${TASK_LIST[@]}"

for TASK in "${TASK_LIST[@]}"; do
  echo
  echo "============================================================"
  echo "TASK: $TASK"
  echo "============================================================"

  ADAPTER_DIR="$MODELS_DIR/$TASK"
  TEST_JSONL="$TASKS_ROOT/$TASK/test.jsonl"

  need_dir "$ADAPTER_DIR"
  need_file "$ADAPTER_DIR/adapter_model.safetensors"
  need_file "$ADAPTER_DIR/adapter_config.json"
  need_file "$TEST_JSONL"

  # -------------------------
  # STEP 1.2: tau_comp stats
  # -------------------------
  echo "=== STEP 1.2: tau_comp stats ==="
  run_py step_1.2.py \
    --adapter_dir "$ADAPTER_DIR" \
    --out "$ART_DIR/${TASK}_tau_comp_stats.pt" \
    --device "$DEVICE"

  # -------------------------
  # STEP 2: layer cosine
  # -------------------------
  echo "=== STEP 2: per-layer cosine ==="
  COS_PT="$ART_DIR/${TASK}_layer_cosine.pt"
  run_py step_2.py \
    --adapter_dir "$ADAPTER_DIR" \
    --base "$BASE_MODEL" \
    --instruct "$INSTRUCT_MODEL" \
    --out "$COS_PT" \
    --device "$DEVICE" \
    --dtype "$DTYPE"

  # -------------------------
  # STEP 3: weights
  # -------------------------
  WEIGHT_FILES=()

  if [[ "$RUN_LINEAR" == "1" ]]; then
    echo "=== STEP 3: weights (linear rank) ==="
    W_JSON="$ART_DIR/${TASK}_weights_linear.json"
    run_py step_3_weights.py --in_pt "$COS_PT" --out_json "$W_JSON" --method linear
    WEIGHT_FILES+=("$W_JSON")
  fi

  if [[ "$RUN_LOG" == "1" ]]; then
    echo "=== STEP 3: weights (log rank) ==="
    W_JSON="$ART_DIR/${TASK}_weights_log.json"
    run_py step_3_weights.py --in_pt "$COS_PT" --out_json "$W_JSON" --method log
    WEIGHT_FILES+=("$W_JSON")
  fi

  if [[ "$RUN_THRESHOLD" == "1" ]]; then
    echo "=== STEP 3: weights (threshold) sigma=$SIGMA ==="
    SIGMA_TAG="$(python - <<PY
s=float("$SIGMA")
print(str(s).replace('.','p').replace('-','m'))
PY
)"
    W_JSON="$ART_DIR/${TASK}_weights_thr_${SIGMA_TAG}.json"
    run_py step_3_weights.py --in_pt "$COS_PT" --out_json "$W_JSON" --method threshold --sigma "$SIGMA"
    WEIGHT_FILES+=("$W_JSON")
  fi

  # -------------------------
  # STEP 4 + STEP 5: apply + eval (Generation/ROUGE)
  # -------------------------
  echo "=== STEP 4+5: apply + eval GEN for lambda=1.0 ==="
  for W_JSON in "${WEIGHT_FILES[@]}"; do
    STRAT="$(basename "$W_JSON")"
    STRAT="${STRAT#${TASK}_weights_}"
    STRAT="${STRAT%.json}"

    OUT_MODEL_DIR="$MERGED_DIR/${TASK}_lata_${STRAT}_l1"
    # Note: Output filename changed to include _gen_preds to distinguish
    OUT_PRED_CSV="$ART_DIR/${TASK}_lata_${STRAT}_l1_gen_preds.csv"

    # Step 4: Merge weights (same as before)
    run_py step_4_apply_lata.py \
      --adapter_dir "$ADAPTER_DIR" \
      --target_model "$INSTRUCT_MODEL" \
      --weights_json "$W_JSON" \
      --lambda_ 1.0 \
      --out_dir "$OUT_MODEL_DIR" \
      --device "$DEVICE" \
      --dtype "$DTYPE"

    # Step 5: Eval using GENERATION script
    run_py step_5_eval_gen.py \
      --model_dir "$OUT_MODEL_DIR" \
      --test_jsonl "$TEST_JSONL" \
      --out_csv "$OUT_PRED_CSV" \
      --dtype "$DTYPE" \
      --max_new_tokens "$MAX_NEW_TOKENS"
  done

  # -------------------------
  # STEP 6: lambda sweep (Generation/ROUGE)
  # -------------------------
  echo "=== STEP 6: lambda sweep GEN (per strategy) ==="
  for W_JSON in "${WEIGHT_FILES[@]}"; do
    STRAT="$(basename "$W_JSON")"
    STRAT="${STRAT#${TASK}_weights_}"
    STRAT="${STRAT%.json}"

    # Note: Output filename changed to include _gen_sweep
    OUT_SWEEP_CSV="$ART_DIR/${TASK}_${STRAT}_gen_sweep.csv"

    run_py step_6_lambda_gen.py \
      --adapter_dir "$ADAPTER_DIR" \
      --weights_json "$W_JSON" \
      --test_jsonl "$TEST_JSONL" \
      --out_csv "$OUT_SWEEP_CSV" \
      --dtype "$DTYPE" \
      --device "$DEVICE" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --lambdas "$LAMBDAS"
  done
done

# =========================
# STEP 8: combine results
# =========================
echo
echo "=== STEP 8: combine all GEN sweeps ==="

python - <<'PY'
import os, glob
import pandas as pd

art = "artifacts"
# Look for the new _gen_sweep files
paths = sorted(glob.glob(os.path.join(art, "*_gen_sweep.csv")))
if not paths:
    print("Warning: No *_gen_sweep.csv files found in artifacts/. Did the sweeps run?")
else:
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["source_csv"] = os.path.basename(p)
        base = os.path.basename(p).replace("_gen_sweep.csv","")
        # heuristic parsing of task/strat
        # format: {TASK}_{STRAT}
        # e.g. agreeableness_low_weights_linear -> task=agreeableness_low, strat=weights_linear
        # But we actually named files: ${TASK}_${STRAT}_gen_sweep.csv
        # where STRAT comes from filename like "weights_linear"
        if "_weights_" in base:
            task, strat = base.split("_weights_", 1)
            strat = "weights_" + strat
        else:
            task, strat = base, "unknown"
            
        df["task"] = task
        df["strategy"] = strat
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(art, "ALL_tasks_ALL_strategies_gen_sweep.csv")
    out.to_csv(out_path, index=False)

    print(f"[ok] wrote: {out_path}")
    # Print preview, sort by task then rouge_l descending
    print(out.sort_values(["task", "rouge_l"], ascending=[True, False]).head(20).to_string(index=False))
PY

echo
echo "DONE"