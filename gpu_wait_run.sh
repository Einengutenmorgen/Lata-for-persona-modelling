#!/usr/bin/env bash
set -euo pipefail

# =========================
# Configuration (defaults)
# =========================
CHECK_INTERVAL=30            # seconds between checks
FREE_DURATION=60             # GPU must satisfy "free" conditions continuously for this long
MEMORY_THRESHOLD_MB=1000     # consider GPU "lightly used" if memory.used < threshold
SCRIPT_TO_RUN="./run_1-8.sh" # script to execute once a GPU is free
REQUIRE_NO_PROCS=1           # 1: require no compute procs on GPU; 0: allow procs (not recommended)
EXCLUDE_PROC_REGEX="ollama"  # if REQUIRE_NO_PROCS=0, still reject GPUs with matching process names

# =========================
# Help
# =========================
usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Waits for a GPU to be "free" for a continuous duration, then runs your script with
CUDA_VISIBLE_DEVICES set to that GPU UUID.

Default "free" condition:
  - memory.used < threshold
  - AND no compute processes on that GPU (recommended)

Options:
  -s, --script PATH         Script to run (default: $SCRIPT_TO_RUN)
  -t, --threshold MB        Memory threshold in MB (default: $MEMORY_THRESHOLD_MB)
  -d, --duration SECS       How long GPU must be free (default: $FREE_DURATION)
  -i, --interval SECS       Check interval (default: $CHECK_INTERVAL)
  --allow-procs             Allow compute processes (NOT recommended). Still excludes by regex.
  --exclude-regex REGEX     Process name regex to exclude when --allow-procs is set (default: $EXCLUDE_PROC_REGEX)
  -h, --help                Show this help

Examples:
  $0 --script ./run_1-8.sh --threshold 500
  $0 --duration 120 --interval 10
  $0 --allow-procs --exclude-regex 'ollama|python'
EOF
  exit 0
}

# =========================
# Arg parsing
# =========================
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--script) SCRIPT_TO_RUN="$2"; shift 2 ;;
    -t|--threshold) MEMORY_THRESHOLD_MB="$2"; shift 2 ;;
    -d|--duration) FREE_DURATION="$2"; shift 2 ;;
    -i|--interval) CHECK_INTERVAL="$2"; shift 2 ;;
    --allow-procs) REQUIRE_NO_PROCS=0; shift 1 ;;
    --exclude-regex) EXCLUDE_PROC_REGEX="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# =========================
# Validation
# =========================
[[ -f "$SCRIPT_TO_RUN" ]] || { echo "ERROR: Script not found: $SCRIPT_TO_RUN"; exit 1; }
[[ -x "$SCRIPT_TO_RUN" ]] || { echo "ERROR: Script not executable: $SCRIPT_TO_RUN"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not found"; exit 1; }

# =========================
# nvidia-smi query helpers
# =========================
get_gpu_memory() {
  # uuid, memory.used (MB)
  nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits
}

get_gpu_procs() {
  # gpu_uuid, pid, process_name, used_memory (MB)
  # If no procs exist, command can output nothing.
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
}

# =========================
# Decide which GPUs are "free"
# =========================
check_free_gpus() {
  local free_gpus=()

  # Build per-UUID flags for compute procs
  declare -A has_any_proc=()
  declare -A has_excluded_proc=()

  while IFS=, read -r uuid pid pname pmem; do
    uuid="$(echo "${uuid:-}" | xargs)"
    pname="$(echo "${pname:-}" | xargs)"
    [[ -z "$uuid" ]] && continue

    has_any_proc["$uuid"]=1
    if [[ "$pname" =~ $EXCLUDE_PROC_REGEX ]]; then
      has_excluded_proc["$uuid"]=1
    fi
  done < <(get_gpu_procs)

  # Evaluate memory + proc conditions
  while IFS=, read -r uuid mem_used; do
    uuid="$(echo "$uuid" | xargs)"
    mem_used="$(echo "$mem_used" | xargs)"

    # memory constraint
    if (( mem_used >= MEMORY_THRESHOLD_MB )); then
      continue
    fi

    # proc constraint
    if (( REQUIRE_NO_PROCS == 1 )); then
      # require no compute processes at all
      [[ -n "${has_any_proc[$uuid]:-}" ]] && continue
    else
      # allow compute procs, but exclude certain names
      [[ -n "${has_excluded_proc[$uuid]:-}" ]] && continue
    fi

    free_gpus+=("$uuid")
  done < <(get_gpu_memory)

  echo "${free_gpus[@]}"
}

# =========================
# Display config
# =========================
echo "=========================================="
echo "GPU Wait-and-Run Script"
echo "=========================================="
echo "Script to run:     $SCRIPT_TO_RUN"
echo "Memory threshold:  ${MEMORY_THRESHOLD_MB} MB"
echo "Free duration:     ${FREE_DURATION} seconds"
echo "Check interval:    ${CHECK_INTERVAL} seconds"
if (( REQUIRE_NO_PROCS == 1 )); then
  echo "Proc rule:         require NO compute processes"
else
  echo "Proc rule:         allow compute processes, exclude regex: $EXCLUDE_PROC_REGEX"
fi
echo "=========================================="
echo
echo "Current GPU status:"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv

# Track free time per GPU UUID
declare -A gpu_free_since

echo
echo "Starting GPU monitoring..."
while true; do
  now_human="$(date '+%Y-%m-%d %H:%M:%S')"
  current_time="$(date +%s)"
  echo
  echo "[$now_human] Checking GPUs..."

  mapfile -t free_gpus < <(check_free_gpus | tr ' ' '\n' | sed '/^$/d')

  if [[ ${#free_gpus[@]} -eq 0 ]]; then
    echo "  No GPUs satisfy conditions (threshold/proc rule)."
    # Reset all timers
    gpu_free_since=()
    echo "  Waiting ${CHECK_INTERVAL}s..."
    sleep "$CHECK_INTERVAL"
    continue
  fi

  echo "  Candidate free GPUs:"
  printf '   - %s\n' "${free_gpus[@]}"

  # Update free timers for currently free GPUs
  for gpu_uuid in "${free_gpus[@]}"; do
    if [[ -z "${gpu_free_since[$gpu_uuid]:-}" ]]; then
      gpu_free_since["$gpu_uuid"]="$current_time"
      echo "  $gpu_uuid: started free period"
    else
      start="${gpu_free_since[$gpu_uuid]}"
      free_time=$(( current_time - start ))
      echo "  $gpu_uuid: free for ${free_time}s (need ${FREE_DURATION}s)"

      if (( free_time >= FREE_DURATION )); then
        echo
        echo "=========================================="
        echo "SUCCESS: GPU $gpu_uuid satisfied conditions for ${free_time}s"
        echo "=========================================="
        echo

        export CUDA_DEVICE_ORDER="PCI_BUS_ID"
        export CUDA_VISIBLE_DEVICES="$gpu_uuid"

        echo "Launching: $SCRIPT_TO_RUN"
        echo "  CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
        echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        echo

        # Optional: show procs/mem right before running
        echo "Pre-launch snapshot:"
        nvidia-smi -i "$CUDA_VISIBLE_DEVICES" || true
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null || true

        "$SCRIPT_TO_RUN"
        exit_code=$?

        echo
        echo "=========================================="
        if [[ $exit_code -eq 0 ]]; then
          echo "Script completed successfully."
        else
          echo "Script failed with exit code: $exit_code"
        fi
        echo "=========================================="
        exit "$exit_code"
      fi
    fi
  done

  # Remove GPUs that are no longer free from tracking
  for gpu_uuid in "${!gpu_free_since[@]}"; do
    still_free=0
    for u in "${free_gpus[@]}"; do
      if [[ "$u" == "$gpu_uuid" ]]; then
        still_free=1
        break
      fi
    done
    if (( still_free == 0 )); then
      echo "  $gpu_uuid: no longer satisfies conditions (was free for $(( current_time - gpu_free_since[$gpu_uuid] ))s)"
      unset gpu_free_since["$gpu_uuid"]
    fi
  done

  echo "  Waiting ${CHECK_INTERVAL}s..."
  sleep "$CHECK_INTERVAL"
done
