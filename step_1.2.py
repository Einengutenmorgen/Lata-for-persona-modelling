#step_1.2.py


import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from safetensors.torch import load_file


TARGET_SUBSTRINGS = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")

def layer_id(name: str) -> int:
    m = LAYER_RE.match(name)
    return int(m.group(1)) if m else -1

def is_target_param(clean_key: str) -> bool:
    return clean_key.startswith("model.layers.") and any(s in clean_key for s in TARGET_SUBSTRINGS)

def get_lora_scaling(adapter_dir: Path) -> float:
    cfg_path = adapter_dir / "adapter_config.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    # scaling used in PEFT LoRA merge: alpha / r
    return float(cfg["lora_alpha"]) / float(cfg["r"])

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True, help="Path to the trained LoRA adapter folder")
    ap.add_argument("--out", default="artifacts/tau_comp_stats.pt", help="Output .pt file")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--matmul_fp32", action="store_true",
                    help="Compute B@A in fp32 (slower, more accurate). Default uses adapter dtype.")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    assert adapter_dir.exists(), f"Missing adapter_dir: {adapter_dir}"
    assert (adapter_dir / "adapter_model.safetensors").exists(), "Missing adapter_model.safetensors"
    assert (adapter_dir / "adapter_config.json").exists(), "Missing adapter_config.json"

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[cfg] device={device} matmul_fp32={args.matmul_fp32}")

    scaling = get_lora_scaling(adapter_dir)
    print(f"[cfg] scaling (alpha/r) = {scaling}")

    # Load LoRA weights
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"), device=device)
    keys = list(tensors.keys())

    # Accumulate on-device to avoid .item() sync in the loop
    per_layer_norm2 = defaultdict(lambda: torch.zeros((), device=device, dtype=torch.float32))
    per_layer_count = defaultdict(int)
    total_norm2 = torch.zeros((), device=device, dtype=torch.float32)

    used = 0
    skipped_non_target = 0
    missing_pairs = 0

    for kA in keys:
        if "lora_A" not in kA or not kA.endswith(".weight"):
            continue

        kB = kA.replace("lora_A", "lora_B")
        if kB not in tensors:
            missing_pairs += 1
            continue

        # Convert LoRA key to base-model weight key
        # Example:
        # base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        # -> model.layers.0.self_attn.q_proj.weight
        clean_key = kA.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")

        if not is_target_param(clean_key):
            skipped_non_target += 1
            continue

        l = layer_id(clean_key)
        if l < 0:
            skipped_non_target += 1
            continue

        A = tensors[kA]
        B = tensors[kB]

        # Dense delta: (B @ A) * (alpha/r)
        if args.matmul_fp32:
            dense = (B.float() @ A.float()) * scaling
        else:
            dense = (B @ A) * scaling

        # Frobenius norm^2
        n2 = (dense.float() * dense.float()).sum()
        per_layer_norm2[l] += n2
        per_layer_count[l] += 1
        total_norm2 += n2
        used += 1

        # free the big dense matrix ASAP
        del dense

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "adapter_dir": str(adapter_dir),
        "scaling_alpha_over_r": scaling,
        "device": device,
        "used_lora_modules": used,            # number of (A,B) pairs used
        "missing_pairs": missing_pairs,
        "skipped_non_target": skipped_non_target,
        "per_layer_norm": {int(l): float(per_layer_norm2[l].sqrt().item()) for l in per_layer_norm2},
        "per_layer_pair_count": dict(per_layer_count),
        "total_norm": float(total_norm2.sqrt().item()),
        "note": "tau_comp for LoRA-from-base equals dense LoRA delta on tuned modules; zero elsewhere.",
    }

    torch.save(result, out_path)
    print(f"[ok] used lora pairs: {used} | missing_pairs: {missing_pairs} | skipped: {skipped_non_target}")
    print(f"[ok] total ||tau_comp||: {result['total_norm']:.6f}")
    print(f"[ok] saved: {out_path}")

if __name__ == "__main__":
    main()
