import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import layer_id, is_target_param


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", required=True, help="HF path to full-param fine-tuned personality model")
    ap.add_argument("--base", required=True, help="HF path to base model (used to compute personality delta)")
    ap.add_argument("--target_model", required=True, help="HF path to model to apply LATA to (typically instruct)")
    ap.add_argument("--weights_json", required=True, help="Layer weights JSON from step_3")
    ap.add_argument("--lambda_", type=float, required=True, help="Global scaling factor")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    w = json.loads(Path(args.weights_json).read_text()).get("layer_weight", {})
    if not w:
        raise RuntimeError("weights_json missing non-empty 'layer_weight'")
    w = {int(k): float(v) for k, v in w.items()}

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    device_map = {"": 0} if device == "cuda" else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print(f"[cfg] device={device} dtype={args.dtype} lambda={args.lambda_}")
    print(f"[load] base         = {args.base}")
    print(f"[load] finetuned    = {args.finetuned}")
    print(f"[load] target_model = {args.target_model}")

    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    ft = AutoModelForCausalLM.from_pretrained(
        args.finetuned, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    base_params = dict(base.named_parameters())
    ft_params = dict(ft.named_parameters())

    updated = 0
    skipped = 0
    upd_norm2 = torch.zeros((), device=device, dtype=torch.float32)

    for name, p in model.named_parameters():
        if not is_target_param(name):
            continue
        if name not in base_params or name not in ft_params:
            skipped += 1
            continue

        l = layer_id(name)
        if l < 0:
            skipped += 1
            continue

        wl = w.get(l)
        if wl is None:
            raise RuntimeError(f"No weight for layer {l} in weights_json")

        delta = (ft_params[name].data - base_params[name].data).float()
        update = (args.lambda_ * wl * delta).to(device=p.device, dtype=p.dtype)

        if p.shape != update.shape:
            raise RuntimeError(f"Shape mismatch at {name}: {tuple(p.shape)} vs {tuple(update.shape)}")

        p.data.add_(update)
        upd_norm2 += (update.float() * update.float()).sum()
        updated += 1
        del delta, update

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ok] updated={updated} skipped={skipped}")
    print(f"[ok] ||applied_update|| = {float(upd_norm2.sqrt().item()):.6f}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[ok] saved merged model to: {out_dir}")


if __name__ == "__main__":
    main()
