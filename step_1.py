#step_1.py
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import argparse
from pathlib import Path
from collections import defaultdict

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM



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

def is_target_param(name: str) -> bool:
    if not name.startswith("model.layers."):
        return False
    return any(s in name for s in TARGET_SUBSTRINGS)

def layer_id(name: str) -> int:
    m = LAYER_RE.match(name)
    if not m:
        return -1
    return int(m.group(1))

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF id/path for base model")
    ap.add_argument("--instruct", required=True, help="HF id/path for instruct model")
    ap.add_argument("--out", default="artifacts/instr_stats.pt", help="output .pt file")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print(f"[load] base     = {args.base}")
    print(f"[load] instruct = {args.instruct}")

    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=dtype, device_map=DEVICE
    )
    inst = AutoModelForCausalLM.from_pretrained(
        args.instruct, dtype=dtype, device_map=DEVICE
    )

    # Fail fast on obvious mismatches
    bc, ic = base.config, inst.config
    assert bc.hidden_size == ic.hidden_size, "hidden_size mismatch"
    assert bc.num_hidden_layers == ic.num_hidden_layers, "num_hidden_layers mismatch"

    base_sd = base.state_dict()
    inst_sd = inst.state_dict()

    common = sorted(set(base_sd.keys()) & set(inst_sd.keys()))
    keys = [k for k in common if is_target_param(k)]
    if not keys:
        raise RuntimeError("No matching target params found. Check model format / key names.")

    per_layer_norm2 = defaultdict(float)
    per_layer_count = defaultdict(int)

    total_norm2 = 0.0
    used = 0

    for k in keys:
        lb = base_sd[k]
        li = inst_sd[k]
        if lb.shape != li.shape:
            raise RuntimeError(f"Shape mismatch at {k}: {lb.shape} vs {li.shape}")

        d = (li - lb).float()  # accumulate in fp32 for stability
        l = layer_id(k)
        if l < 0:
            continue

        n2 = torch.sum(d * d).item()
        per_layer_norm2[l] += n2
        per_layer_count[l] += 1
        total_norm2 += n2
        used += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "base": args.base,
        "instruct": args.instruct,
        "target_param_substrings": list(TARGET_SUBSTRINGS),
        "used_param_count": used,
        "per_layer_norm": {int(l): (per_layer_norm2[l] ** 0.5) for l in per_layer_norm2},
        "per_layer_param_count": dict(per_layer_count),
        "total_norm": total_norm2 ** 0.5,
        "dtype_accumulated": "float32",
    }

    torch.save(result, out_path)
    print(f"[ok] used params: {used}")
    print(f"[ok] total ||tau_instr||: {result['total_norm']:.6f}")
    print(f"[ok] saved: {out_path}")

if __name__ == "__main__":
    main()
