#step_5_eval.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import json
import argparse
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

AB_RE = re.compile(r"\b([AB])\b", re.IGNORECASE)

def extract_ab(text: str):
    if text is None:
        return None
    m = AB_RE.search(text.strip())
    if not m:
        return None
    return m.group(1).upper()

def build_prompt(example: dict) -> str:
    # Your jsonl already provides chat messages: system + user (+ assistant label).
    # For evaluation we want the model to answer, so we include system + user only.

    # old from TASK_II
    msgs = example["messages"]
    sys = next((m["content"] for m in msgs if m["role"] == "system"), "")
    user = next((m["content"] for m in msgs if m["role"] == "user"), "")

    #pompt=example['prompt']


    # Minimal: concatenate with clear separators
    return f"{pompt}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Merged model dir (HF format)")
    ap.add_argument("--test_jsonl", required=True, help="Path to test.jsonl")
    ap.add_argument("--out_csv", default="artifacts/preds.csv")
    ap.add_argument("--max_new_tokens", type=int, default=3)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device_map = {"": 0} if torch.cuda.is_available() else "cpu"

    print(f"[load] model={args.model_dir}")
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    model.eval()

    # Ensure pad token exists
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = []
    correct = 0
    total = 0

    test_path = Path(args.test_jsonl)
    assert test_path.exists(), f"Missing: {test_path}"

    with test_path.open("r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"[data] loaded {len(data)} examples")

    for ex in tqdm(data):
        prompt = build_prompt(ex)
        gold = ex.get("correct_option", None)
        if gold is not None:
            gold = gold.strip().upper()

        inputs = tok(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

        decoded = tok.decode(out[0], skip_special_tokens=True)
        gen = decoded[len(prompt):].strip()
        pred = extract_ab(gen)

        is_ok = (pred == gold) if (pred is not None and gold is not None) else False
        if gold in ("A", "B"):
            total += 1
            if is_ok:
                correct += 1

        rows.append({
            "id": ex.get("id"),
            "gold": gold,
            "pred": pred,
            "generated": gen,
        })

    acc = correct / total if total > 0 else 0.0
    print(f"[ok] evaluated={total}  correct={correct}  accuracy={acc:.4f}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[ok] wrote: {out_csv}")

if __name__ == "__main__":
    main()
