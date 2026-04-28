import argparse
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from utils import layer_id, is_target_param


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", required=True, help="HF path to full-param fine-tuned personality model")
    ap.add_argument("--base", required=True, help="HF path to base model")
    ap.add_argument("--instruct", required=True, help="HF path to instruct model")
    ap.add_argument("--out", default="artifacts/layer_cosine.pt")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    device_map = {"": 0} if device == "cuda" else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print(f"[cfg] device={device} dtype={args.dtype}")
    print(f"[load] base      = {args.base}")
    print(f"[load] instruct  = {args.instruct}")
    print(f"[load] finetuned = {args.finetuned}")

    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    inst = AutoModelForCausalLM.from_pretrained(
        args.instruct, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    ft = AutoModelForCausalLM.from_pretrained(
        args.finetuned, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )

    bc, ic = base.config, inst.config
    assert bc.hidden_size == ic.hidden_size, "hidden_size mismatch"
    assert bc.num_hidden_layers == ic.num_hidden_layers, "num_hidden_layers mismatch"

    base_params = dict(base.named_parameters())
    inst_params = dict(inst.named_parameters())
    ft_params = dict(ft.named_parameters())

    dot_layer = defaultdict(lambda: torch.zeros((), device=device, dtype=torch.float32))
    n2_comp_layer = defaultdict(lambda: torch.zeros((), device=device, dtype=torch.float32))
    n2_instr_layer = defaultdict(lambda: torch.zeros((), device=device, dtype=torch.float32))

    used = 0
    skipped = 0

    for wname in base_params:
        if not is_target_param(wname):
            continue
        if wname not in inst_params or wname not in ft_params:
            skipped += 1
            continue

        l = layer_id(wname)
        if l < 0:
            skipped += 1
            continue

        tau_comp = (ft_params[wname].data - base_params[wname].data).float()
        tau_instr = (inst_params[wname].data - base_params[wname].data).float()

        dot_layer[l] += (tau_comp * tau_instr).sum()
        n2_comp_layer[l] += (tau_comp * tau_comp).sum()
        n2_instr_layer[l] += (tau_instr * tau_instr).sum()
        used += 1

        del tau_comp, tau_instr

    cos = {}
    for l in sorted(set(dot_layer) | set(n2_comp_layer) | set(n2_instr_layer)):
        a = dot_layer[l].item()
        na = n2_comp_layer[l].sqrt().item()
        nb = n2_instr_layer[l].sqrt().item()
        cos[l] = a / (na * nb) if (na > 0 and nb > 0) else 0.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "base": args.base,
        "instruct": args.instruct,
        "finetuned": args.finetuned,
        "dtype": args.dtype,
        "device": device,
        "used_params": used,
        "skipped_params": skipped,
        "layer_cosine": cos,
        "note": "Cosine between tau_comp (finetuned-base) and tau_instr (instruct-base) per layer.",
    }

    torch.save(result, out_path)

    top = sorted(cos.items(), key=lambda x: x[1], reverse=True)[:5]
    bot = sorted(cos.items(), key=lambda x: x[1])[:5]
    print(f"[ok] used={used} skipped={skipped}")
    print("[cos] top 5:", top)
    print("[cos] bot 5:", bot)
    print(f"[ok] saved: {out_path}")


if __name__ == "__main__":
    main()
