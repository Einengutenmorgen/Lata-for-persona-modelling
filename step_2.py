#step_2.py
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from safetensors.torch import load_file
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

def layer_id(name: str) -> int:
    m = LAYER_RE.match(name)
    return int(m.group(1)) if m else -1

def is_target_param(name: str) -> bool:
    return name.startswith("model.layers.") and any(s in name for s in TARGET_SUBSTRINGS)

def get_lora_scaling(adapter_dir: Path) -> float:
    cfg_path = adapter_dir / "adapter_config.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    return float(cfg["lora_alpha"]) / float(cfg["r"])

def clean_lora_key(kA: str) -> str:
    # Common PEFT key patterns:
    # base_model.model.model.layers.0....lora_A.weight
    # base_model.model.layers.0....lora_A.weight
    # We remove "base_model.model." prefix once; remaining should start with "model.layers..."
    k = kA
    if k.startswith("base_model.model."):
        k = k.replace("base_model.model.", "", 1)
    return k.replace(".lora_A.weight", ".weight")

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True, help="Path to LoRA adapter folder")
    ap.add_argument("--base", required=True, help="HF id/path for base model")
    ap.add_argument("--instruct", required=True, help="HF id/path for instruct model")
    ap.add_argument("--out", default="artifacts/layer_cosine.pt", help="Output .pt file")

    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--matmul_fp32", action="store_true",
                    help="Compute LoRA dense (B@A) in fp32 (slower, more accurate).")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    assert adapter_dir.exists(), f"Missing adapter_dir: {adapter_dir}"
    assert (adapter_dir / "adapter_model.safetensors").exists(), "Missing adapter_model.safetensors"
    assert (adapter_dir / "adapter_config.json").exists(), "Missing adapter_config.json"

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    device_map = {"": 0} if device == "cuda" else "cpu"

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print(f"[cfg] device={device} device_map={device_map} weights_dtype={args.dtype}")
    print(f"[load] base     = {args.base}")
    print(f"[load] instruct = {args.instruct}")

    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    inst = AutoModelForCausalLM.from_pretrained(
        args.instruct, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )

    # fail fast config checks
    bc, ic = base.config, inst.config
    assert bc.hidden_size == ic.hidden_size, "hidden_size mismatch"
    assert bc.num_hidden_layers == ic.num_hidden_layers, "num_hidden_layers mismatch"

    # Build lightweight name->Parameter maps (keeps tensors on-device)
    base_params = dict(base.named_parameters())
    inst_params = dict(inst.named_parameters())

    # Load LoRA tensors on-device
    scaling = get_lora_scaling(adapter_dir)
    print(f"[cfg] lora scaling alpha/r = {scaling}")

    lora = load_file(str(adapter_dir / "adapter_model.safetensors"), device=device)
    lora_keys = list(lora.keys())

    # Per-layer accumulators (float32 tensors on same device)
    dot_layer = defaultdict(lambda: torch.zeros((), device=device, dtype=torch.float32))
    n2_comp_layer = defaultdict(lambda: torch.zeros((), device=device, dtype=torch.float32))
    n2_instr_layer = defaultdict(lambda: torch.zeros((), device=device, dtype=torch.float32))

    used = 0
    missing_pairs = 0
    missing_model_param = 0
    skipped_non_target = 0

    for kA in lora_keys:
        if "lora_A" not in kA or not kA.endswith(".weight"):
            continue
        kB = kA.replace("lora_A", "lora_B")
        if kB not in lora:
            missing_pairs += 1
            continue

        wname = clean_lora_key(kA)
        if not is_target_param(wname):
            skipped_non_target += 1
            continue

        if wname not in base_params or wname not in inst_params:
            # PEFT naming mismatch or different model class
            missing_model_param += 1
            continue

        l = layer_id(wname)
        if l < 0:
            skipped_non_target += 1
            continue

        A = lora[kA]
        B = lora[kB]

        # tau_comp on tuned weights: dense LoRA delta
        if args.matmul_fp32:
            tau_comp = (B.float() @ A.float()) * scaling
        else:
            tau_comp = (B @ A) * scaling

        # tau_instr: instruct - base on same weight
        w_base = base_params[wname].data
        w_inst = inst_params[wname].data
        tau_instr = (w_inst - w_base).float()

        # Accumulate dot and norms in fp32
        tc = tau_comp.float()
        di = tau_instr  # already fp32
        dot_layer[l] += (tc * di).sum()
        n2_comp_layer[l] += (tc * tc).sum()
        n2_instr_layer[l] += (di * di).sum()

        used += 1
        del tau_comp, tc, di, tau_instr

    # Compute cosine per layer on CPU (final scalars)
    cos = {}
    for l in sorted(set(dot_layer.keys()) | set(n2_comp_layer.keys()) | set(n2_instr_layer.keys())):
        a = dot_layer[l].item()
        na = n2_comp_layer[l].sqrt().item()
        nb = n2_instr_layer[l].sqrt().item()
        if na == 0.0 or nb == 0.0:
            cos[l] = 0.0
        else:
            cos[l] = a / (na * nb)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "adapter_dir": str(adapter_dir),
        "base": args.base,
        "instruct": args.instruct,
        "weights_dtype": args.dtype,
        "device": device,
        "scaling_alpha_over_r": scaling,
        "used_lora_pairs": used,
        "missing_pairs": missing_pairs,
        "missing_model_param": missing_model_param,
        "skipped_non_target": skipped_non_target,
        "layer_cosine": cos,
        "note": "Cosine computed between tau_comp (LoRA dense delta) and tau_instr (instruct-base) on tuned projection weights.",
    }

    torch.save(result, out_path)

    # quick visibility
    print(f"[ok] used={used} missing_pairs={missing_pairs} missing_model_param={missing_model_param} skipped={skipped_non_target}")
    top = sorted(cos.items(), key=lambda x: x[1], reverse=True)[:5]
    bot = sorted(cos.items(), key=lambda x: x[1])[:5]
    print("[cos] top 5 most-instruction-like layers:", top)
    print("[cos] bot 5 least-instruction-like layers:", bot)
    print(f"[ok] saved: {out_path}")

if __name__ == "__main__":
    main()
