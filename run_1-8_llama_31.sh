#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIGURATION
# =========================

# GPU Setup
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# Models
BASE_MODEL="meta-llama/Llama-3.1-8B"
INSTRUCT_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Paths
MODELS_DIR="/home/s2chhauu/lata/models_llama31_8b_sft"
TASKS_ROOT="${TASKS_ROOT:-/home/s2chhauu/lata/Task_III}" 
ART_DIR="artifacts_llama31"
MERGED_DIR="models_seq_llama31"

# Runtime
DTYPE="bfloat16" 
DEVICE="cuda"
MAX_NEW_TOKENS="128" 
LAMBDAS="0.0,0.25,0.5,1.0,2.0"

# Strategies
RUN_LINEAR="1"
RUN_LOG="1"
RUN_THRESHOLD="1"
SIGMA="0.0002"

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
need_file "step_1.py"
need_file "step_1.2.py"
need_file "step_2.py"
need_file "step_3_weights.py"
need_file "step_4_apply_lata.py"
need_file "step_5_eval_gen.py"
need_file "step_6_lambda_gen.py"

need_dir "$MODELS_DIR"
need_dir "$TASKS_ROOT"

mkdir -p "$ART_DIR" "$MERGED_DIR"

# =========================
# STEP 1 (global): instr stats
# =========================
echo "=== STEP 1: instruction vector stats (global) ==="
if [[ -f "$ART_DIR/instr_stats.pt" ]]; then
    echo " [SKIP] Found $ART_DIR/instr_stats.pt"
else
    run_py step_1.py \
      --base "$BASE_MODEL" \
      --instruct "$INSTRUCT_MODEL" \
      --out "$ART_DIR/instr_stats.pt" \
      --dtype "$DTYPE"
fi

# =========================
# Iterate tasks
# =========================
TASK_LIST=()
while IFS= read -r -d '' d; do
  TASK_LIST+=("$(basename "$d")")
done < <(find "$TASKS_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

[[ "${#TASK_LIST[@]}" -gt 0 ]] || die "No task directories found under: $TASKS_ROOT"

echo
echo "Found tasks: ${TASK_LIST[*]}"

for TASK in "${TASK_LIST[@]}"; do
  echo
  echo "============================================================"
  echo "TASK: $TASK"
  echo "============================================================"

  ADAPTER_DIR="$MODELS_DIR/$TASK"
  TEST_JSONL="$TASKS_ROOT/$TASK/test.jsonl"

  if [[ ! -d "$ADAPTER_DIR" ]]; then
    echo "WARNING: Model directory not found for task $TASK. Skipping."
    continue
  fi

  need_file "$ADAPTER_DIR/adapter_model.safetensors"
  need_file "$ADAPTER_DIR/adapter_config.json"
  need_file "$TEST_JSONL"

  # -------------------------
  # STEP 1.2: tau_comp stats
  # -------------------------
  echo "=== STEP 1.2: tau_comp stats ==="
  OUT_FILE="$ART_DIR/${TASK}_tau_comp_stats.pt"
  if [[ -f "$OUT_FILE" ]]; then
      echo " [SKIP] Found $OUT_FILE"
  else
      run_py step_1.2.py \
        --adapter_dir "$ADAPTER_DIR" \
        --out "$OUT_FILE" \
        --device "$DEVICE"
  fi

  # -------------------------
  # STEP 2: layer cosine
  # -------------------------
  echo "=== STEP 2: per-layer cosine ==="
  COS_PT="$ART_DIR/${TASK}_layer_cosine.pt"
  if [[ -f "$COS_PT" ]]; then
      echo " [SKIP] Found $COS_PT"
  else
      run_py step_2.py \
        --adapter_dir "$ADAPTER_DIR" \
        --base "$BASE_MODEL" \
        --instruct "$INSTRUCT_MODEL" \
        --out "$COS_PT" \
        --device "$DEVICE" \
        --dtype "$DTYPE"
  fi

  # -------------------------
  # STEP 3: weights
  # -------------------------
  WEIGHT_FILES=()

  # Linear
  if [[ "$RUN_LINEAR" == "1" ]]; then
    W_JSON="$ART_DIR/${TASK}_weights_linear.json"
    if [[ ! -f "$W_JSON" ]]; then
        echo "=== STEP 3: weights (linear) ==="
        run_py step_3_weights.py --in_pt "$COS_PT" --out_json "$W_JSON" --method linear
    fi
    WEIGHT_FILES+=("$W_JSON")
  fi

  # Log
  if [[ "$RUN_LOG" == "1" ]]; then
    W_JSON="$ART_DIR/${TASK}_weights_log.json"
    if [[ ! -f "$W_JSON" ]]; then
        echo "=== STEP 3: weights (log) ==="
        run_py step_3_weights.py --in_pt "$COS_PT" --out_json "$W_JSON" --method log
    fi
    WEIGHT_FILES+=("$W_JSON")
  fi

  # Threshold
  if [[ "$RUN_THRESHOLD" == "1" ]]; then
    SIGMA_TAG="$(python - <<PY
s=float("$SIGMA")
print(str(s).replace('.','p').replace('-','m'))
PY
)"
    W_JSON="$ART_DIR/${TASK}_weights_thr_${SIGMA_TAG}.json"
    if [[ ! -f "$W_JSON" ]]; then
        echo "=== STEP 3: weights (threshold) ==="
        run_py step_3_weights.py --in_pt "$COS_PT" --out_json "$W_JSON" --method threshold --sigma "$SIGMA"
    fi
    WEIGHT_FILES+=("$W_JSON")
  fi

  # -------------------------
  # STEP 4 + STEP 5: apply + eval (WITH AUTO-DELETE)
  # -------------------------
  echo "=== STEP 4+5: apply + eval GEN ==="
  for W_JSON in "${WEIGHT_FILES[@]}"; do
    STRAT="$(basename "$W_JSON")"
    STRAT="${STRAT#${TASK}_weights_}"
    STRAT="${STRAT%.json}"

    OUT_MODEL_DIR="$MERGED_DIR/${TASK}_lata_${STRAT}_l1"
    OUT_PRED_CSV="$ART_DIR/${TASK}_lata_${STRAT}_l1_gen_preds.csv"

    # RESUME CHECK: If the CSV exists, we are done with this model.
    if [[ -f "$OUT_PRED_CSV" ]]; then
        echo " [SKIP] Results exist for $STRAT. Skipping generation."
        # Safety cleanup in case the folder was left behind
        if [[ -d "$OUT_MODEL_DIR" ]]; then
            echo " [CLEANUP] Removing old model dir $OUT_MODEL_DIR"
            rm -rf "$OUT_MODEL_DIR"
        fi
        continue
    fi

    # 1. Create Model
    run_py step_4_apply_lata.py \
      --adapter_dir "$ADAPTER_DIR" \
      --target_model "$INSTRUCT_MODEL" \
      --weights_json "$W_JSON" \
      --lambda_ 1.0 \
      --out_dir "$OUT_MODEL_DIR" \
      --device "$DEVICE" \
      --dtype "$DTYPE"

    # 2. Eval
    run_py step_5_eval_gen.py \
      --model_dir "$OUT_MODEL_DIR" \
      --test_jsonl "$TEST_JSONL" \
      --out_csv "$OUT_PRED_CSV" \
      --dtype "$DTYPE" \
      --max_new_tokens "$MAX_NEW_TOKENS"

    # 3. CRITICAL CLEANUP: Delete the model immediately to free disk space
    echo " [CLEANUP] Deleting temporary model: $OUT_MODEL_DIR"
    rm -rf "$OUT_MODEL_DIR"
  done

  # -------------------------
  # STEP 6: lambda sweep
  # -------------------------
  echo "=== STEP 6: lambda sweep GEN ==="
  for W_JSON in "${WEIGHT_FILES[@]}"; do
    STRAT="$(basename "$W_JSON")"
    STRAT="${STRAT#${TASK}_weights_}"
    STRAT="${STRAT%.json}"

    OUT_SWEEP_CSV="$ART_DIR/${TASK}_${STRAT}_gen_sweep.csv"

    if [[ -f "$OUT_SWEEP_CSV" ]]; then
        echo " [SKIP] Sweep exists for $STRAT"
        continue
    fi

    run_py step_6_lambda_gen.py \
      --adapter_dir "$ADAPTER_DIR" \
      --base_model "$BASE_MODEL" \
      --instruct_model "$INSTRUCT_MODEL" \
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

python - <<PY
import os, glob
import pandas as pd

art = "$ART_DIR"
paths = sorted(glob.glob(os.path.join(art, "*_gen_sweep.csv")))
if not paths:
    print("Warning: No *_gen_sweep.csv files found.")
else:
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["source_csv"] = os.path.basename(p)
        base = os.path.basename(p).replace("_gen_sweep.csv","")
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
PY

echo
echo "DONE"