#!/usr/bin/env python3
# step_9_big5_inventory.py
#
# Runs a 50-item Big Five inventory with multiple local/HF models.
# Refactored to use standard Text Generation (Completion) templates 
# instead of Chat Templates.

import argparse
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
)

# 1..5 standalone digit (accepts "1", "(1)", " 1 ", etc.)
RE_CLASS = re.compile(r"(?<!\d)([1-5])(?!\d)")


def parse_likert_1_to_5(text: str):
    if not text:
        return None

    m = RE_CLASS.search(text)
    if m:
        return int(m.group(1))

    t = text.strip().lower()
    # basic word fallback (keeps it minimal)
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
    # stable across runs/machines (unlike Python's built-in hash)
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def load_questionnaire(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data[0]
    task = block["task"]
    questions = block["questions"]
    return task, questions


def build_prompt(tokenizer, task: str, statement: str):
    """
    Refactored: Uses a raw text completion template.
    This works for both Base and Instruct models by treating the task 
    as a document completion.
    """
    # We ignore tokenizer.chat_template and build a raw string.
    # We explicitly define the scale and format.
    
    prompt = f"""{task}

Rate how much you agree with the statement below on a scale from 1 to 5.
1: Strongly Disagree
2: Disagree
3: Neutral
4: Agree
5: Strongly Agree

Reply with exactly one number.

Statement: "{statement}"

Answer:"""
    return prompt


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str, seed: int, device: str, max_new_tokens: int):
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

    # Slice the output to get only the new tokens
    gen = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def infer_llama_config_if_missing(model_path: str):
    """
    Fix for: local folders missing 'model_type' in config.json.
    """
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return None

    cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    if "model_type" in cfg_dict and cfg_dict["model_type"]:
        return None

    arch = cfg_dict.get("architectures", []) or []
    is_llama = any("llama" in a.lower() for a in arch) or any("llama" in str(k).lower() for k in cfg_dict.keys())

    if not is_llama:
        return None

    cfg_dict["model_type"] = "llama"
    return LlamaConfig(**cfg_dict)


def try_load_as_peft_adapter(model_path: str, device: str, torch_dtype):
    """
    If model_path is a PEFT adapter directory, load base + adapter.
    """
    adapter_cfg = Path(model_path) / "adapter_config.json"
    if not adapter_cfg.exists():
        return None

    try:
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError(
            f"Detected PEFT adapter at '{model_path}' but 'peft' is not installed."
        )

    cfg = json.loads(adapter_cfg.read_text(encoding="utf-8"))
    base_name = cfg.get("base_model_name_or_path")
    if not base_name:
        raise RuntimeError(
            f"PEFT adapter at '{model_path}' missing 'base_model_name_or_path'"
        )

    tok = AutoTokenizer.from_pretrained(base_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        dtype=torch_dtype,
        device_map=None,
    ).to(device)
    base.eval()

    model = PeftModel.from_pretrained(base, model_path).to(device)
    model.eval()
    return model, tok


def load_model(model_id_or_path: str, device: str, dtype: str):
    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]

    peft_loaded = try_load_as_peft_adapter(model_id_or_path, device, dtype)
    if peft_loaded is not None:
        return peft_loaded

    tok = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    cfg_override = None
    if Path(model_id_or_path).exists():
        cfg_override = infer_llama_config_if_missing(model_id_or_path)

    mdl = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        dtype=dtype,
        device_map=None,
        config=cfg_override,
    ).to(device)
    mdl.eval()
    return mdl, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questionnaire", type=str, required=True)
    ap.add_argument("--out", type=str, required=True, help="Output prefix (no extension)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    # Reduced max tokens slightly as we expect just a digit
    ap.add_argument("--max_new_tokens", type=int, default=4) 
    args = ap.parse_args()

    task, questions = load_questionnaire(Path(args.questionnaire))

    models = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B-Instruct",
        #"models/agreeableness_low",
        # "merged/agreeableness_high_lata_linear_l1",
        # "merged/agreeableness_high_lata_log_l1",
        # "merged/agreeableness_high_lata_thr_0p0002_l1"
        # "merged/agreeableness_low_lata_linear_l1",
        # "merged/agreeableness_low_lata_log_l1",
        # "merged/agreeableness_low_lata_thr_0p0002_l1"
        "models_high_r8/agreeableness_high",
        "models_high_r8/agreeableness_low",
        "models_personality_sft/agreeableness_high",
        "models_personality_sft/agreeableness_low",

    ]

    detailed_rows = []
    wide_rows = []

    for model_name in models:
        print(f"Loading {model_name}...")
        try:
            model, tok = load_model(model_name, args.device, args.dtype)
        except Exception as e:
            print(f"Skipping {model_name} due to error: {e}")
            continue
            
        model_seed_base = args.seed + stable_int(model_name)

        for run in range(1, args.repeats + 1):
            wide = {"model": model_name}

            for q in questions:
                qid = int(q["id"])
                stmt = q["content"]
                dim = q.get("dimension", "")

                prompt = build_prompt(tok, task, stmt)
                
                # Generate
                raw = generate_one(
                    model=model,
                    tokenizer=tok,
                    prompt=prompt,
                    seed=model_seed_base + run * 1000 + qid,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                )
                
                extracted = parse_likert_1_to_5(raw)
                
                # Debug print for first item to ensure formatting works
                if qid == 1 and run == 1:
                    print(f"--- Prompt Example ({model_name}) ---\n{prompt}\n--- Output: {raw} -> {extracted} ---")

                wide[f"X_{qid}"] = extracted

                detailed_rows.append(
                    {
                        "model": model_name,
                        "run": run,
                        "question_id": qid,
                        "dimension": dim,
                        "statement": stmt,
                        "prompt": prompt,
                        "raw_answer": raw,
                        "extracted_answer": extracted,
                        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
                    }
                )

            wide_rows.append(wide)

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    details_df = pd.DataFrame(detailed_rows)

    wide_df = pd.DataFrame(wide_rows)
    x_cols = [c for c in wide_df.columns if c.startswith("X_")]
    for c in x_cols:
        wide_df[c] = pd.to_numeric(wide_df[c], errors="coerce").astype("Int64")

    out_prefix = Path(args.out)
    details_path = out_prefix.with_suffix(".details.csv")
    responses_path = out_prefix.with_suffix(".responses.csv")

    details_df.to_csv(details_path, index=False)
    wide_df.to_csv(responses_path, index=False)

    print(f"Wrote:\n  {details_path}\n  {responses_path}")


if __name__ == "__main__":
    main()