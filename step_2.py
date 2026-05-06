#step_2.py
import argparse
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from utils import layer_id, is_target_param


@torch.no_grad()
def extract_task_vector(model_path: str, base_sd: dict, dtype: torch.dtype) -> dict:
    """
    Load a model, compute (model - base) per target parameter, return as a
    CPU float32 dict, then immediately free the model from GPU memory.
    """
    print(f"[load] {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    sd = model.state_dict()
    task_vector = {}
    for k, base_w in base_sd.items():
        if not is_target_param(k):
            continue
        if k not in sd:
            continue
        l = layer_id(k)
        if l < 0:
            continue
        task_vector[k] = (sd[k].float() - base_w.to(sd[k].device).float()).cpu()

    del model, sd
    torch.cuda.empty_cache()
    return task_vector


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--base",      required=True)
    ap.add_argument("--instruct",  required=True)
    ap.add_argument("--out",       default="artifacts/layer_cosine.pt")
    ap.add_argument("--dtype",     default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device",    default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    dtype = {"float16": torch.float16,
             "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    # ------------------------------------------------------------------
    # 1. Load base once — kept on CPU as float32 reference
    # ------------------------------------------------------------------
    print(f"[load] base = {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    base_sd = {k: v.float().cpu() for k, v in base.state_dict().items()
               if is_target_param(k) and layer_id(k) >= 0}
    del base
    torch.cuda.empty_cache()
    print(f"[base] kept {len(base_sd)} target params on CPU")

    # ------------------------------------------------------------------
    # 2. Extract task vectors sequentially — only one extra model in GPU
    #    memory at a time
    # ------------------------------------------------------------------
    tau_instr = extract_task_vector(args.instruct,  base_sd, dtype)
    tau_comp  = extract_task_vector(args.finetuned, base_sd, dtype)

    # base_sd no longer needed
    del base_sd

    # ------------------------------------------------------------------
    # 3. Accumulate cosine components per layer entirely on CPU
    # ------------------------------------------------------------------
    dot_layer      = defaultdict(float)
    n2_comp_layer  = defaultdict(float)
    n2_instr_layer = defaultdict(float)
    used = skipped = 0

    for k in tau_instr:
        if k not in tau_comp:
            skipped += 1
            continue
        l = layer_id(k)
        tc = tau_comp[k]
        ti = tau_instr[k]
        dot_layer[l]      += (tc * ti).sum().item()
        n2_comp_layer[l]  += (tc * tc).sum().item()
        n2_instr_layer[l] += (ti * ti).sum().item()
        used += 1

    cos = {}
    for l in sorted(set(dot_layer)):
        a  = dot_layer[l]
        na = n2_comp_layer[l]  ** 0.5
        nb = n2_instr_layer[l] ** 0.5
        if na == 0.0 and nb == 0.0:
            # Both deltas are zero — layer unchanged in both models.
            # Treat as perfectly aligned (not orthogonal).
            cos[l] = 1.0
        elif na == 0.0 or nb == 0.0:
            # One delta is zero — undefined cosine, conservatively treat as aligned.
            cos[l] = 1.0
        else:
            cos[l] = a / (na * nb)

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "base":         args.base,
        "instruct":     args.instruct,
        "finetuned":    args.finetuned,
        "dtype":        args.dtype,
        "used_params":  used,
        "skipped_params": skipped,
        "layer_cosine": cos,
        "note": "Cosine between tau_comp (finetuned-base) and tau_instr (instruct-base) per layer.",
    }
    torch.save(result, out_path)

    top = sorted(cos.items(), key=lambda x: x[1], reverse=True)[:5]
    bot = sorted(cos.items(), key=lambda x: x[1])[:5]
    print(f"[ok] used={used}  skipped={skipped}")
    print(f"[cos] top 5: {top}")
    print(f"[cos] bot 5: {bot}")
    print(f"[ok] saved: {out_path}")


if __name__ == "__main__":
    main()