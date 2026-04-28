#!/usr/bin/env python3
# Runs the 50-item IPIP Big Five inventory across one or more models.
# Pass models via --models (comma-separated paths/HF IDs) or --models_json (JSON file).

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RE_CLASS = re.compile(r"(?<!\d)([1-5])(?!\d)")


def parse_likert_1_to_5(text: str):
    if not text:
        return None
    m = RE_CLASS.search(text)
    if m:
        return int(m.group(1))
    t = text.strip().lower()
    if "strongly disagree" in t:
        return 1
    if "strongly agree" in t:
        return 5
    if re.search(r"\bdisagree\b", t):
        return 2
    if re.search(r"\bagree\b", t):
        return 4
    if re.search(r"\bneutral\b", t) or re.search(r"\bneither\b", t):
        return 3
    return None


def stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def load_questionnaire(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data[0]
    return block["task"], block["questions"]


def build_prompt(task: str, statement: str) -> str:
    return (
        f"{task}\n\n"
        "Rate how much you agree with the statement below on a scale from 1 to 5.\n"
        "1: Strongly Disagree\n2: Disagree\n3: Neutral\n4: Agree\n5: Strongly Agree\n\n"
        "Reply with exactly one number.\n\n"
        f'Statement: "{statement}"\n\nAnswer:'
    )


def load_model(model_id_or_path: str, device: str, torch_dtype):
    tok = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id_or_path, torch_dtype=torch_dtype, device_map=None
    ).to(device)
    mdl.eval()
    return mdl, tok


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str, seed: int, device: str, max_new_tokens: int) -> str:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questionnaire", required=True)
    ap.add_argument("--out", required=True, help="Output path prefix (no extension)")
    ap.add_argument("--models", default="",
                    help="Comma-separated model paths/HF IDs to evaluate")
    ap.add_argument("--models_json", default="",
                    help="Path to a JSON file containing a list of model paths")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_new_tokens", type=int, default=4)
    args = ap.parse_args()

    if not args.models and not args.models_json:
        ap.error("Provide --models or --models_json")

    if args.models_json:
        model_list = json.loads(Path(args.models_json).read_text())
    else:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if args.device == "cpu":
        torch_dtype = torch.float32

    task, questions = load_questionnaire(Path(args.questionnaire))

    detailed_rows = []
    wide_rows = []

    for model_name in model_list:
        print(f"\n[load] {model_name}")
        try:
            model, tok = load_model(model_name, args.device, torch_dtype)
        except Exception as e:
            print(f"[skip] {model_name}: {e}")
            continue

        model_seed_base = args.seed + stable_int(model_name)

        for run in range(1, args.repeats + 1):
            wide = {"model": model_name}

            for q in questions:
                qid = int(q["id"])
                stmt = q["content"]
                dim = q.get("dimension", "")
                prompt = build_prompt(task, stmt)

                raw = generate_one(
                    model=model,
                    tokenizer=tok,
                    prompt=prompt,
                    seed=model_seed_base + run * 1000 + qid,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                )
                extracted = parse_likert_1_to_5(raw)

                if qid == 1 and run == 1:
                    print(f"[sample] prompt:\n{prompt}\n→ raw={raw!r}  extracted={extracted}")

                wide[f"X_{qid}"] = extracted
                detailed_rows.append({
                    "model": model_name,
                    "run": run,
                    "question_id": qid,
                    "dimension": dim,
                    "statement": stmt,
                    "raw_answer": raw,
                    "extracted_answer": extracted,
                    "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
                })

            wide_rows.append(wide)

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    details_df = pd.DataFrame(detailed_rows)
    wide_df = pd.DataFrame(wide_rows)
    for c in [col for col in wide_df.columns if col.startswith("X_")]:
        wide_df[c] = pd.to_numeric(wide_df[c], errors="coerce").astype("Int64")

    out_prefix = Path(args.out)
    details_df.to_csv(out_prefix.with_suffix(".details.csv"), index=False)
    wide_df.to_csv(out_prefix.with_suffix(".responses.csv"), index=False)
    print(f"\n[ok] wrote {out_prefix.with_suffix('.details.csv')} and {out_prefix.with_suffix('.responses.csv')}")


if __name__ == "__main__":
    main()
