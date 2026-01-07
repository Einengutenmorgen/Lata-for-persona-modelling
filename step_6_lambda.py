#step_6_lambda.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import pandas as pd
from tqdm import tqdm
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

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
AB_RE = re.compile(r"\b([AB])\b", re.IGNORECASE)

def extract_ab(text: str):
    if text is None:
        return None
    m = AB_RE.search(text.strip())
    return m.group(1).upper() if m else None

def build_prompt(example: dict) -> str:
    # Your jsonl already provides chat messages: system + user (+ assistant label).
    # For evaluation we want the model to answer, so we include system + user only.

    # old from TASK_II
    # msgs = example["messages"]
    # sys = next((m["content"] for m in msgs if m["role"] == "system"), "")
    # user = next((m["content"] for m in msgs if m["role"] == "user"), "")

    pompt=example['prompt']


    # Minimal: concatenate with clear separators
    return f"{pompt}"

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
    # Handle common PEFT prefixes
    k = kA
    if k.startswith("base_model.model."):
        k = k.replace("base_model.model.", "", 1)
    return k.replace(".lora_A.weight", ".weight")

def load_tokenizer(model_id_or_path: str):
    # Work around occasional tokenizer warnings; harmless if unsupported
    try:
        return AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)

@torch.no_grad()
def eval_model(model, tok, data, max_new_tokens: int):
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    correct = 0
    total = 0

    for ex in tqdm(data, leave=False):
        prompt = build_prompt(ex)
        
        # gold = ex.get("correct_option", None)
        # gold = gold.strip().upper() if isinstance(gold, str) else None

        gold = ex.get('chosen', None)
        print(f'Prompt: {prompt} \n Gold: {gold}')

        inputs = tok(prompt, return_tensors="pt").to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

        decoded = tok.decode(out[0], skip_special_tokens=True)
        gen = decoded[len(prompt):].strip()
        pred = extract_ab(gen)

        if gold in ("A", "B"):
            total += 1
            if pred == gold:
                correct += 1

    acc = correct / total if total else 0.0
    return {"total": total, "correct": correct, "accuracy": acc}

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--weights_json", required=True)

    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--instruct_model", default="meta-llama/Llama-3.2-1B-Instruct")

    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--lambdas", default="0.0,0.25,0.5,1.0,2.0")
    ap.add_argument("--out_csv", default="artifacts/step7_lambda_sweep.csv")

    ap.add_argument("--max_new_tokens", type=int, default=3)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--matmul_fp32", action="store_true")

    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    assert adapter_dir.exists(), f"Missing adapter_dir: {adapter_dir}"
    assert (adapter_dir / "adapter_model.safetensors").exists(), "Missing adapter_model.safetensors"
    assert (adapter_dir / "adapter_config.json").exists(), "Missing adapter_config.json"

    wobj = json.loads(Path(args.weights_json).read_text())
    w = wobj.get("layer_weight", None)
    if not isinstance(w, dict) or not w:
        raise RuntimeError("weights_json must contain non-empty 'layer_weight'")
    w = {int(k): float(v) for k, v in w.items()}

    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    device_map = {"": 0} if device == "cuda" else "cpu"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    # Load data once
    test_path = Path(args.test_jsonl)
    assert test_path.exists(), f"Missing: {test_path}"
    with test_path.open("r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"[data] loaded {len(data)} examples")

    results = []

    # Baseline: BASE
    print(f"[baseline] base model: {args.base_model}")
    tok_base = load_tokenizer(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    base_model.eval()
    r = eval_model(base_model, tok_base, data, args.max_new_tokens)
    results.append({"run": "base", "lambda": None, **r})
    del base_model
    torch.cuda.empty_cache()

    # Baseline: INSTRUCT
    print(f"[baseline] instruct model: {args.instruct_model}")
    tok_inst = load_tokenizer(args.instruct_model)
    inst_model = AutoModelForCausalLM.from_pretrained(
        args.instruct_model, dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    inst_model.eval()

    r = eval_model(inst_model, tok_inst, data, args.max_new_tokens)
    results.append({"run": "instruct", "lambda": None, **r})

    # Precompute weighted deltas for the 112 tuned weights
    print("[prep] loading LoRA + building weighted deltas")
    scaling = get_lora_scaling(adapter_dir)
    lora = load_file(str(adapter_dir / "adapter_model.safetensors"), device=device)
    params = dict(inst_model.named_parameters())

    # Cache original values + weighted deltas (on GPU)
    cache = []  # list of (param, orig_tensor, weighted_delta_tensor)
    missing_pairs = 0
    missing_param = 0
    used = 0

    for kA in lora.keys():
        if "lora_A" not in kA or not kA.endswith(".weight"):
            continue
        kB = kA.replace("lora_A", "lora_B")
        if kB not in lora:
            missing_pairs += 1
            continue

        wname = clean_lora_key(kA)
        if not is_target_param(wname):
            continue
        if wname not in params:
            missing_param += 1
            continue

        l = layer_id(wname)
        if l < 0:
            continue
        wl = w.get(l, None)
        if wl is None:
            raise RuntimeError(f"No weight for layer {l} in weights_json")

        A = lora[kA]
        B = lora[kB]

        if args.matmul_fp32:
            dense = (B.float() @ A.float()) * scaling
        else:
            dense = (B @ A) * scaling

        # store delta already scaled by layer weight
        wd = (wl * dense).to(device=params[wname].device, dtype=params[wname].dtype)
        orig = params[wname].data.clone()  # save original once
        cache.append((params[wname], orig, wd))
        used += 1
        del dense

    print(f"[prep] used={used} missing_pairs={missing_pairs} missing_param={missing_param}")
    if used == 0:
        raise RuntimeError("No LoRA deltas were applied. Check key cleaning and target modules.")

    # Lambda sweep (in-place update + restore each time)
    for lam in lambdas:
        print(f"[sweep] lambda={lam}")
        for p, orig, wd in cache:
            # p = orig + lam * wd
            p.data.copy_(orig + (lam * wd))
        r = eval_model(inst_model, tok_inst, data, args.max_new_tokens)
        results.append({"run": "lata_linear", "lambda": lam, **r})

    # Write summary
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"[ok] wrote: {out_csv}")
    print(df)

if __name__ == "__main__":
    main()
