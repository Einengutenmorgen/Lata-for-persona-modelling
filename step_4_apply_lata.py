#step_4_apply_lata.py
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import json
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    # Typical PEFT patterns:
    # base_model.model.model.layers.0....lora_A.weight
    # base_model.model.layers.0....lora_A.weight
    k = kA
    if k.startswith("base_model.model."):
        k = k.replace("base_model.model.", "", 1)
    return k.replace(".lora_A.weight", ".weight")

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--target_model", required=True, help="HF id/path for target model (use instruct here)")
    ap.add_argument("--weights_json", required=True, help="Output from step4 (contains layer_weight)")
    ap.add_argument("--lambda_", type=float, required=True, help="Scaling for update")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--matmul_fp32", action="store_true", help="Compute B@A in fp32 (slower, more accurate)")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    assert adapter_dir.exists(), f"Missing adapter_dir: {adapter_dir}"
    assert (adapter_dir / "adapter_model.safetensors").exists(), "Missing adapter_model.safetensors"
    assert (adapter_dir / "adapter_config.json").exists(), "Missing adapter_config.json"

    weights_obj = json.loads(Path(args.weights_json).read_text())
    w = weights_obj.get("layer_weight", None)
    if w is None or not isinstance(w, dict) or len(w) == 0:
        raise RuntimeError("weights_json missing non-empty 'layer_weight' dict")
    w = {int(k): float(v) for k, v in w.items()}

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    device_map = {"": 0} if device == "cuda" else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print(f"[cfg] device={device} device_map={device_map} weights_dtype={args.dtype}")
    print(f"[cfg] lambda={args.lambda_}")
    print(f"[load] target_model={args.target_model}")

    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        dtype=dtype,                # NOTE: uses new arg name
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    params = dict(model.named_parameters())

    scaling = get_lora_scaling(adapter_dir)
    print(f"[cfg] lora scaling alpha/r = {scaling}")

    lora = load_file(str(adapter_dir / "adapter_model.safetensors"), device=device)
    lora_keys = list(lora.keys())

    updated = 0
    missing_pairs = 0
    missing_param = 0
    skipped = 0

    # Track update energy (optional sanity)
    upd_norm2 = torch.zeros((), device=device, dtype=torch.float32)

    for kA in lora_keys:
        if "lora_A" not in kA or not kA.endswith(".weight"):
            continue
        kB = kA.replace("lora_A", "lora_B")
        if kB not in lora:
            missing_pairs += 1
            continue

        wname = clean_lora_key(kA)
        if not is_target_param(wname):
            skipped += 1
            continue

        if wname not in params:
            missing_param += 1
            continue

        l = layer_id(wname)
        if l < 0:
            skipped += 1
            continue

        wl = w.get(l, None)
        if wl is None:
            raise RuntimeError(f"weights_json has no weight for layer {l}")

        A = lora[kA]
        B = lora[kB]

        # dense LoRA delta
        if args.matmul_fp32:
            delta = (B.float() @ A.float()) * scaling
        else:
            delta = (B @ A) * scaling

        update = (args.lambda_ * wl) * delta

        p = params[wname]
        if p.shape != update.shape:
            raise RuntimeError(f"Shape mismatch {wname}: param={tuple(p.shape)} update={tuple(update.shape)}")

        # ensure update on same device/dtype as param
        update = update.to(device=p.device, dtype=p.dtype)
        p.data.add_(update)

        upd_norm2 += (update.float() * update.float()).sum()

        updated += 1
        del delta, update

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ok] updated_pairs={updated} missing_pairs={missing_pairs} missing_param={missing_param} skipped={skipped}")
    print(f"[ok] ||applied_update|| = {float(upd_norm2.sqrt().item()):.6f}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[ok] saved merged model to: {out_dir}")

if __name__ == "__main__":
    main()
