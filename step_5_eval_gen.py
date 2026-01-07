# step_5_eval_gen.py
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_csv", default="artifacts/gen_preds.csv")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device_map = {"": 0} if torch.cuda.is_available() else "cpu"

    print(f"[load] model={args.model_dir}")
    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, fix_mistral_regex=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
        
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    model.eval()

    # Initialize ROUGE Scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    test_path = Path(args.test_jsonl)
    with test_path.open("r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"[data] loaded {len(data)} examples")

    rows = []
    total_score = 0.0
    count = 0

    for ex in tqdm(data):
        prompt = ex.get("prompt", "")
        target = ex.get("chosen", "")
        if not target:
            continue

        # Direct generation (No Chat Template)
        prompt_str = prompt

        inputs = tok(prompt_str, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

        decoded_all = tok.decode(out[0], skip_special_tokens=True)
        
        # Robustly extract new tokens only
        input_len = len(tok.decode(inputs.input_ids[0], skip_special_tokens=True))
        gen = decoded_all[input_len:].strip()

        # Calculate ROUGE-L
        scores = scorer.score(target, gen)
        score = scores['rougeL'].fmeasure
        
        total_score += score
        count += 1

        rows.append({
            "id": ex.get("id"),
            "prompt": prompt,
            "target_chosen": target,
            "generated": gen,
            "rouge_l": score
        })

    avg_score = total_score / count if count > 0 else 0.0
    print(f"[ok] evaluated={count}  avg_rouge_l={avg_score:.4f}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[ok] wrote: {out_csv}")

if __name__ == "__main__":
    main()