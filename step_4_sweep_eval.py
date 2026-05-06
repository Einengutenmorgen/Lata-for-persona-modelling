# step_4_sweep_eval.py
# Lambda sweep: evaluates base, instruct, and LATA-steered model variants
# using cosine similarity between generated text and reference answers.
#
# Base and instruct baselines are cached in the output CSV — re-running for
# a new finetuned model skips those evaluations automatically.

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from utils import layer_id, is_target_param, load_tokenizer


@torch.no_grad()
def get_embedding(text: str, model, tok) -> torch.Tensor:
    """
    Mean-pool the last hidden states over non-padding tokens.
    Returns a normalised 1-D float32 vector on CPU.
    """
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    out    = model(**inputs, output_hidden_states=True)
    # last hidden state: (1, seq_len, hidden_size)
    hidden = out.hidden_states[-1]
    mask   = inputs["attention_mask"].unsqueeze(-1).float()
    emb    = (hidden * mask).sum(dim=1) / mask.sum(dim=1)   # mean pool
    return F.normalize(emb.squeeze(0).float().cpu(), dim=0)


@torch.no_grad()
def eval_model(model, tok, data, max_new_tokens, embed_model, embed_tok):
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    total_score = 0.0
    count = 0

    for ex in tqdm(data, leave=False, desc="eval"):
        prompt = ex.get("prompt", "")
        target = ex.get("answer", "")
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
        input_len   = len(tok.decode(inputs.input_ids[0], skip_special_tokens=True))
        gen         = decoded_all[input_len:].strip()

        target_emb = get_embedding(target, embed_model, embed_tok)
        gen_emb    = get_embedding(gen,    embed_model, embed_tok)
        score      = (target_emb * gen_emb).sum().item()   # both normalised → dot = cosine

        total_score += score
        count += 1

    return {"count": count, "cosine_sim": total_score / count if count > 0 else 0.0}


def load_existing_results(out_csv: Path) -> tuple[list[dict], set[str]]:
    """
    Load previously saved results from CSV.
    Returns (rows_as_dicts, set_of_completed_run_labels).
    A run label is 'base' or 'instruct' — lata rows are always re-run
    because they depend on the finetuned model.
    """
    if not out_csv.exists():
        return [], set()
    df = pd.read_csv(out_csv)
    rows = df.to_dict("records")
    completed = {r["run"] for r in rows if r["run"] in ("base", "instruct")}
    return rows, completed


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned",      required=True)
    ap.add_argument("--base_model",     required=True)
    ap.add_argument("--instruct_model", required=True)
    ap.add_argument("--weights_json",   required=True)
    ap.add_argument("--test_jsonl",     required=True)
    ap.add_argument("--lambdas",        default="0.0,0.5,1.0,1.5,2.0")
    ap.add_argument("--out_csv",        default="artifacts/sweep_eval.csv")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--subset_size",    type=int, default=50)
    ap.add_argument("--dtype",          default="bfloat16")
    ap.add_argument("--device",         default="cuda")
    args = ap.parse_args()

    print("[init] using model hidden states for embedding (no external embedder)")
    w          = {int(k): float(v) for k, v in
                  json.loads(Path(args.weights_json).read_text()).get("layer_weight", {}).items()}
    lambdas    = [float(x) for x in args.lambdas.split(",") if x.strip()]
    dtype      = getattr(torch, args.dtype)
    device_map = {"": 0} if args.device == "cuda" else "cpu"
    out_csv    = Path(args.out_csv)

    # ---- data -----------------------------------------------------------
    with Path(args.test_jsonl).open() as f:
        data = [json.loads(l) for l in f if l.strip()]
    if args.subset_size and len(data) > args.subset_size:
        data = random.sample(data, args.subset_size)
    print(f"[data] {len(data)} examples")

    for col in ("prompt", "answer"):
        if col not in data[0]:
            raise KeyError(f"test_jsonl missing column '{col}'. Found: {list(data[0].keys())}")

    # ---- checkpoint: reload any already-completed baseline rows ---------
    results, completed = load_existing_results(out_csv)
    if completed:
        print(f"[cache] skipping already-completed baselines: {sorted(completed)}")

    # ---- instruct model loads first — used as fixed embedder for all runs --
    # This ensures cosine scores are comparable across base / instruct / lata.
    print(f"\n[load] instruct = {args.instruct_model}  (fixed embedder for all conditions)")
    embed_tok   = load_tokenizer(args.instruct_model)
    embed_model = AutoModelForCausalLM.from_pretrained(
        args.instruct_model, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    embed_model.eval()

    # ---- baseline: base model (skip if cached) --------------------------
    if "base" not in completed:
        print(f"\n[run] base = {args.base_model}")
        tok   = load_tokenizer(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
        )
        model.eval()
        results.append({"run": "base", "lambda": None,
                        **eval_model(model, tok, data, args.max_new_tokens, embed_model, embed_tok)})
        del model
        torch.cuda.empty_cache()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print("[cache] base result saved")
    else:
        print(f"\n[skip] base already in {out_csv}")

    # ---- baseline: instruct model (generation + embedding are same model) -
    if "instruct" not in completed:
        print(f"\n[run] instruct = {args.instruct_model}")
        results.append({"run": "instruct", "lambda": None,
                        **eval_model(embed_model, embed_tok, data, args.max_new_tokens, embed_model, embed_tok)})
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print("[cache] instruct result saved")
    else:
        print(f"\n[skip] instruct already in {out_csv}")

    # ---- precompute deltas (base and ft on CPU) -------------------------
    print(f"\n[prep] computing deltas: {args.finetuned}")
    base_cpu = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="cpu", low_cpu_mem_usage=True
    )
    ft_cpu = AutoModelForCausalLM.from_pretrained(
        args.finetuned, torch_dtype=dtype, device_map="cpu", low_cpu_mem_usage=True
    )
    base_sd = dict(base_cpu.named_parameters())
    ft_sd   = dict(ft_cpu.named_parameters())

    cache = []
    used  = 0
    for name, p in embed_model.named_parameters():
        if not is_target_param(name):
            continue
        if name not in base_sd or name not in ft_sd:
            continue
        l  = layer_id(name)
        wl = w.get(l)
        if l < 0 or wl is None:
            continue
        delta = (ft_sd[name].data - base_sd[name].data).to(device=p.device, dtype=p.dtype)
        cache.append((p, p.data.clone().cpu(), wl * delta))
        used += 1

    del base_cpu, ft_cpu, base_sd, ft_sd
    torch.cuda.empty_cache()
    print(f"[prep] cached {used} parameter deltas")

    if used == 0:
        raise RuntimeError("No parameters matched. Check --finetuned / --base_model paths.")

    # ---- lambda sweep ---------------------------------------------------
    results = [r for r in results if r["run"] != "lata"]

    # Move orig weights to GPU once — avoids repeated PCIe transfers per lambda.
    cache_gpu = [(p, orig.to(p.device), wd) for p, orig, wd in cache]
    del cache

    for lam in lambdas:
        print(f"\n[sweep] lambda = {lam}")
        for p, orig, wd in cache_gpu:
            p.data.copy_(orig + lam * wd)
        results.append({"run": "lata", "lambda": lam,
                        **eval_model(embed_model, embed_tok, data, args.max_new_tokens, embed_model, embed_tok)})

    # restore instruct weights
    for p, orig, _ in cache_gpu:
        p.data.copy_(orig)

    # ---- save final -----------------------------------------------------
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\n[ok] saved: {out_csv}")
    print(df[["run", "lambda", "cosine_sim"]].to_string(index=False))


if __name__ == "__main__":
    main()