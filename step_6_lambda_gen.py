import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from utils import layer_id, is_target_param, load_tokenizer


@torch.no_grad()
def eval_model(model, tok, data, max_new_tokens, scorer):
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    total_score = 0.0
    count = 0

    for ex in tqdm(data, leave=False, desc="Eval"):
        prompt = ex.get("prompt", "")
        target = ex.get("chosen", "")
        if not target:
            continue

        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

        decoded_all = tok.decode(out[0], skip_special_tokens=True)
        input_len = len(tok.decode(inputs.input_ids[0], skip_special_tokens=True))
        gen = decoded_all[input_len:].strip()

        score = scorer.score(target, gen)["rougeL"].fmeasure
        total_score += score
        count += 1

    return {"count": count, "rouge_l": total_score / count if count > 0 else 0.0}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", required=True, help="HF path to full-param fine-tuned personality model")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--instruct_model", required=True)
    ap.add_argument("--weights_json", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--lambdas", default="0.0,0.5,1.0,2.0")
    ap.add_argument("--out_csv", default="artifacts/sweep_gen.csv")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    w = {int(k): float(v) for k, v in
         json.loads(Path(args.weights_json).read_text()).get("layer_weight", {}).items()}
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    dtype = getattr(torch, args.dtype)
    device = args.device
    device_map = {"": 0} if device == "cuda" else "cpu"

    with Path(args.test_jsonl).open("r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"[data] {len(data)} examples")

    results = []

    # Baseline: base model
    print(f"[run] base model: {args.base_model}")
    tok = load_tokenizer(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    model.eval()
    results.append({"run": "base", "lambda": None, **eval_model(model, tok, data, args.max_new_tokens, scorer)})
    del model
    torch.cuda.empty_cache()

    # Baseline: instruct model (LATA is applied to this)
    print(f"[run] instruct model: {args.instruct_model}")
    tok = load_tokenizer(args.instruct_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.instruct_model, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    model.eval()
    results.append({"run": "instruct", "lambda": None, **eval_model(model, tok, data, args.max_new_tokens, scorer)})

    # Precompute personality deltas: w[layer] * (finetuned - base)
    print(f"[prep] loading base + finetuned to compute deltas: {args.finetuned}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.finetuned, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    base_params = dict(base_model.named_parameters())
    ft_params = dict(ft_model.named_parameters())
    inst_params = dict(model.named_parameters())

    # cache: list of (param_on_instruct, original_value, weighted_delta)
    cache = []
    used = 0

    for name, p in inst_params.items():
        if not is_target_param(name):
            continue
        if name not in base_params or name not in ft_params:
            continue

        l = layer_id(name)
        if l < 0:
            continue
        wl = w.get(l)
        if wl is None:
            continue

        delta = (ft_params[name].data - base_params[name].data).to(device=p.device, dtype=p.dtype)
        wd = wl * delta
        cache.append((p, p.data.clone(), wd))
        used += 1
        del delta

    del base_model, ft_model
    torch.cuda.empty_cache()
    print(f"[prep] cached deltas for {used} parameters")

    if used == 0:
        raise RuntimeError("No parameters matched. Check --finetuned and --base_model paths.")

    # Lambda sweep: apply, evaluate, restore
    for lam in lambdas:
        print(f"[sweep] lambda={lam}")
        for p, orig, wd in cache:
            p.data.copy_(orig + lam * wd)
        results.append({"run": "lata", "lambda": lam, **eval_model(model, tok, data, args.max_new_tokens, scorer)})

    # Restore original instruct weights
    for p, orig, _ in cache:
        p.data.copy_(orig)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[ok] wrote: {out_csv}")
    print(pd.DataFrame(results)[["run", "lambda", "rouge_l"]])


if __name__ == "__main__":
    main()
