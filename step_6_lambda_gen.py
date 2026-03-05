# step_6_lambda_gen.py
import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import pandas as pd
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer

TARGET_SUBSTRINGS = (
    ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
)
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")

def layer_id(name: str):
    m = LAYER_RE.match(name)
    return int(m.group(1)) if m else -1

def is_target_param(name: str):
    return name.startswith("model.layers.") and any(s in name for s in TARGET_SUBSTRINGS)

def clean_lora_key(k):
    if k.startswith("base_model.model."): k = k.replace("base_model.model.", "", 1)
    return k.replace(".lora_A.weight", ".weight")

def get_lora_scaling(adapter_dir):
    with open(adapter_dir / "adapter_config.json", "r") as f: cfg = json.load(f)
    return float(cfg["lora_alpha"]) / float(cfg["r"])

def load_tokenizer(path):
    try: return AutoTokenizer.from_pretrained(path, use_fast=True, fix_mistral_regex=True)
    except: return AutoTokenizer.from_pretrained(path, use_fast=True)

@torch.no_grad()
def eval_model(model, tok, data, max_new_tokens, scorer):
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    total_score = 0.0
    count = 0
    
    for ex in tqdm(data, leave=False, desc="Eval"):
        prompt = ex.get("prompt", "")
        target = ex.get("chosen", "")
        if not target: continue

        # Direct generation (No Chat Template)
        prompt_str = prompt

        inputs = tok(prompt_str, return_tensors="pt").to(model.device)
        
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id
        )
        
        decoded_all = tok.decode(out[0], skip_special_tokens=True)
        input_len = len(tok.decode(inputs.input_ids[0], skip_special_tokens=True))
        gen = decoded_all[input_len:].strip()

        # ROUGE-L
        scores = scorer.score(target, gen)
        score = scores['rougeL'].fmeasure
        
        total_score += score
        count += 1

    avg = total_score / count if count > 0 else 0.0
    return {"count": count, "rouge_l": avg}

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--weights_json", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--instruct_model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--lambdas", default="0.0,0.5,1.0,2.0")
    ap.add_argument("--out_csv", default="artifacts/sweep_gen.csv")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--matmul_fp32", action="store_true")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    assert adapter_dir.exists()
    
    # Initialize Scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    wobj = json.loads(Path(args.weights_json).read_text())
    w = {int(k): float(v) for k, v in wobj.get("layer_weight", {}).items()}

    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    dtype = getattr(torch, args.dtype)
    device = args.device

    with Path(args.test_jsonl).open("r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"[data] {len(data)} examples")

    results = []

    # 1. Base
    print(f"[run] Base Model: {args.base_model}")
    tok = load_tokenizer(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype, device_map={"":0})
    model.eval()
    res = eval_model(model, tok, data, args.max_new_tokens, scorer)
    results.append({"run": "base", "lambda": None, **res})
    del model
    torch.cuda.empty_cache()

    # 2. Instruct (Baseline)
    print(f"[run] Instruct Model: {args.instruct_model}")
    tok = load_tokenizer(args.instruct_model)
    model = AutoModelForCausalLM.from_pretrained(args.instruct_model, dtype=dtype, device_map={"":0})
    model.eval()
    
    res = eval_model(model, tok, data, args.max_new_tokens, scorer)
    results.append({"run": "instruct", "lambda": None, **res})

    # 3. Prep Deltas
    print("[prep] Computing LoRA deltas...")
    scaling = get_lora_scaling(adapter_dir)
    lora = load_file(str(adapter_dir / "adapter_model.safetensors"), device=device)
    params = dict(model.named_parameters())
    
    cache = []
    used = 0
    
    for kA in lora.keys():
        if "lora_A" not in kA or not kA.endswith(".weight"): continue
        kB = kA.replace("lora_A", "lora_B")
        if kB not in lora: continue
        
        wname = clean_lora_key(kA)
        if not is_target_param(wname) or wname not in params: continue
        
        l = layer_id(wname)
        wl = w.get(l, None)
        if wl is None: continue 

        A = lora[kA]
        B = lora[kB]
        
        if args.matmul_fp32:
            dense = (B.float() @ A.float()) * scaling
        else:
            dense = (B @ A) * scaling
            
        wd = (wl * dense).to(device=params[wname].device, dtype=params[wname].dtype)
        orig = params[wname].data.clone()
        cache.append((params[wname], orig, wd))
        used += 1
        del dense
        
    print(f"[prep] Cached updates for {used} modules")

    # 4. Sweep
    for lam in lambdas:
        print(f"[sweep] lambda={lam}")
        for p, orig, wd in cache:
            p.data.copy_(orig + (lam * wd))
        
        res = eval_model(model, tok, data, args.max_new_tokens, scorer)
        results.append({"run": "lata_gen", "lambda": lam, **res})

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[ok] wrote: {out_csv}")
    print(pd.DataFrame(results)[["run", "lambda", "rouge_l"]])

if __name__ == "__main__":
    main()