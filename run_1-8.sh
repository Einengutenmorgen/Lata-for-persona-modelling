#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIG (edit as needed)
# =========================

# GPU selection (optional)
#export CUDA_VISIBLE_DEVICES="3"
# Replace your hardcoded export with this:
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1


# Verify immediately - this will print to your terminal so you can be 100% sure
echo "------------------------------------------------"
echo "DEBUG: CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"
echo "DEBUG: NVIDIA-SMI for this specific ID:"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"
python - <<'PY'
import os, torch
print("IN-SCRIPT CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

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
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-120}"
LAMBDAS="${LAMBDAS:-0.0,0.25,0.5,1.0,2.0}"

# Strategies to run (linear, log, threshold)
RUN_LINEAR="${RUN_LINEAR:-1}"
RUN_LOG="${RUN_LOG:-1}"
RUN_THRESHOLD="${RUN_THRESHOLD:-1}"

# Threshold sigma (cosines in your run were ~1e-4; 0.95 would drop nothing)
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
need_file "step_1.py"
need_file "step_1.2.py"
need_file "step_2.py"
need_file "step_3_weights.py"
need_file "step_4_apply_lata.py"
need_file "step_5_eval.py"
need_file "step_6_lambda.py"
need_file "step_8_combine.py"

need_dir "$MODELS_DIR"
need_dir "$TASKS_ROOT"

mkdir -p "$ART_DIR" "$MERGED_DIR"

python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print("torch cuda:0 name:", p.name)
print("torch cuda:0 pci_bus_id:", getattr(p, "pci_bus_id", None))
print("torch cuda:0 uuid:", getattr(p, "uuid", None))
free, total = torch.cuda.mem_get_info(0)
print("torch cuda:0 mem:", round(free/1024**3,2), "GiB free /", round(total/1024**3,2), "GiB total")
PY


# =========================
# STEP 1 (global): instr stats
# =========================
echo "=== STEP 1: instruction vector stats (global) ==="
run_py step_1.py \
  --base "$BASE_MODEL" \
  --instruct "$INSTRUCT_MODEL" \
  --out "$ART_DIR/instr_stats.pt"

# =========================
# Iterate tasks (each subdir in Task_II)
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
  # STEP 1.2: tau_comp stats (LoRA delta)
  # -------------------------
  echo "=== STEP 1.2: tau_comp stats ==="
  run_py step_1.2.py \
    --adapter_dir "$ADAPTER_DIR" \
    --out "$ART_DIR/${TASK}_tau_comp_stats.pt" \
    --device "$DEVICE"

  # -------------------------
  # STEP 2: layer cosine (tau_comp vs tau_instr)
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
  # STEP 3: weights (strategies)
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
    # Make sigma filename safe
    SIGMA_TAG="$(python - <<PY
s=float("$SIGMA")
print(str(s).replace('.','p').replace('-','m'))
PY
)"
    W_JSON="$ART_DIR/${TASK}_weights_thr_${SIGMA_TAG}.json"
    run_py step_3_weights.py --in_pt "$COS_PT" --out_json "$W_JSON" --method threshold --sigma "$SIGMA"
    WEIGHT_FILES+=("$W_JSON")
  fi

  [[ "${#WEIGHT_FILES[@]}" -gt 0 ]] || die "No strategies enabled. Set RUN_LINEAR/RUN_LOG/RUN_THRESHOLD."

  # -------------------------
  # STEP 4 + STEP 5: apply + eval (default lambda=1.0)
  # -------------------------
  echo "=== STEP 4+5: apply + eval for lambda=1.0 (per strategy) ==="
  for W_JSON in "${WEIGHT_FILES[@]}"; do
    STRAT="$(basename "$W_JSON")"
    STRAT="${STRAT#${TASK}_weights_}"
    STRAT="${STRAT%.json}"

    OUT_MODEL_DIR="$MERGED_DIR/${TASK}_lata_${STRAT}_l1"
    OUT_PRED_CSV="$ART_DIR/${TASK}_lata_${STRAT}_l1_preds.csv"

    run_py step_4_apply_lata.py \
      --adapter_dir "$ADAPTER_DIR" \
      --target_model "$INSTRUCT_MODEL" \
      --weights_json "$W_JSON" \
      --lambda_ 1.0 \
      --out_dir "$OUT_MODEL_DIR" \
      --device "$DEVICE" \
      --dtype "$DTYPE"

    run_py step_5_eval.py \
      --model_dir "$OUT_MODEL_DIR" \
      --test_jsonl "$TEST_JSONL" \
      --out_csv "$OUT_PRED_CSV" \
      --dtype "$DTYPE" \
      --max_new_tokens "$MAX_NEW_TOKENS"
  done

  # -------------------------
  # STEP 6: lambda sweep (per strategy)
  # -------------------------
  echo "=== STEP 6: lambda sweep (per strategy) ==="
  for W_JSON in "${WEIGHT_FILES[@]}"; do
    STRAT="$(basename "$W_JSON")"
    STRAT="${STRAT#${TASK}_weights_}"
    STRAT="${STRAT%.json}"

    OUT_SWEEP_CSV="$ART_DIR/${TASK}_${STRAT}_lambda_sweep.csv"

    run_py step_6_lambda.py \
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
# STEP 8: combine results (per task)
# If your step_8_combine.py expects fixed filenames, call it manually instead.
# Here we combine sweeps in a generic way into one CSV.
# =========================
echo
echo "=== STEP 8: combine all sweeps into one table ==="

python - <<'PY'
import os, glob
import pandas as pd

art = "artifacts"
paths = sorted(glob.glob(os.path.join(art, "*_lambda_sweep.csv")))
if not paths:
    raise SystemExit("No *_lambda_sweep.csv files found in artifacts/")

dfs = []
for p in paths:
    df = pd.read_csv(p)
    df["source_csv"] = os.path.basename(p)
    # try to parse task + strategy from filename: <task>_<strategy>_lambda_sweep.csv
    base = os.path.basename(p).replace("_lambda_sweep.csv","")
    if "_" in base:
        task, strat = base.rsplit("_", 1)
    else:
        task, strat = base, "unknown"
    df["task"] = task
    df["strategy"] = strat
    dfs.append(df)

out = pd.concat(dfs, ignore_index=True)
out_path = os.path.join(art, "ALL_tasks_ALL_strategies_lambda_sweep.csv")
out.to_csv(out_path, index=False)

print(f"[ok] wrote: {out_path}")
print(out.sort_values(["task","strategy","run","lambda"], na_position="first").head(50).to_string(index=False))
PY

echo
echo "DONE"
