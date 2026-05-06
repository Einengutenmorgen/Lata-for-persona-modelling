# finetune_personality_sft.py
# Full-parameter SFT — NO LoRA.
# Loss is masked to assistant tokens only (next-token prediction on target).
# Required for downstream LATA task-vector computation (θ_ft - θ_base).
#
# Expects per-profile:
#   {PROCESSED_ROOT}/{profile}/train.jsonl   columns: prompt, answer, trait, level
#   {PROCESSED_ROOT}/{profile}/dev.jsonl
#
# Saves full model weights to:
#   {OUTPUT_ROOT}/{profile}/

import gc
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

# =========================
# HUGGINGFACE LOGIN
# =========================

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    sys.exit("ERROR: HF_TOKEN environment variable not set.")
login(token=HF_TOKEN)

# =========================
# HARD-CODED CONFIG
# =========================

MODEL_NAME_OR_PATH = "meta-llama/Llama-3.1-8B-Instruct"
PROCESSED_ROOT     = "profiles/"               # output of build_profiles.py
OUTPUT_ROOT        = "/media/data/hau/models_llama31_8b_full_ft"
WANDB_ID           = "llama_31_8b_profiles"

PROFILES = (
    "Resilient",
    "Overcontrolled",
    "Undercontrolled",
)

# Training hyperparameters
SEED                    = 42
EPOCHS                  = 1
BATCH_SIZE              = 1          # per-device
GRAD_ACCUMULATION_STEPS = 8          # effective batch = 8
LEARNING_RATE           = 2e-5
WEIGHT_DECAY            = 0.01
MAX_LENGTH              = 512
WARMUP_RATIO            = 0.03
LOGGING_STEPS           = 100
SAVE_TOTAL_LIMIT        = 2

# Generic system message — personality signal is in the training examples,
# NOT in the prompt, so the model internalises the trait into its weights.
SYSTEM_MSG = "Answer in first person!"

REPORT_TO = "wandb"   # "wandb" or "none"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def fail_fast_env_checks(tokenizer: AutoTokenizer) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer has no eos_token.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def to_prompt_completion(example: dict, tokenizer: AutoTokenizer) -> dict:
    """
    Convert a row to the TRL prompt/completion format.
    SFTTrainer concatenates these before tokenisation and masks the prompt
    tokens from the loss when completion_only_loss=True.
    """
    prompt_messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": example["prompt"].rstrip()},
    ]
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    completion = example["answer"].rstrip() + tokenizer.eos_token
    return {"prompt": prompt, "completion": completion}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    fail_fast_env_checks(tokenizer)

    for profile in PROFILES:
        train_path = os.path.join(PROCESSED_ROOT, profile, "train.jsonl")
        dev_path   = os.path.join(PROCESSED_ROOT, profile, "dev.jsonl")
        require_file(train_path)
        require_file(dev_path)

        out_dir = os.path.join(OUTPUT_ROOT, profile)
        os.makedirs(out_dir, exist_ok=True)

        if REPORT_TO == "wandb":
            os.environ["WANDB_NAME"] = f"sft_{WANDB_ID}_{profile}"
            os.environ["WANDB_TAGS"] = f"sft,{WANDB_ID},{profile}"

        # ---- load & validate ------------------------------------------------
        dataset = load_dataset(
            "json",
            data_files={"train": train_path, "validation": dev_path},
            keep_in_memory=False,
        )

        for split in ("train", "validation"):
            missing = {"prompt", "answer"} - set(dataset[split].column_names)
            if missing:
                raise ValueError(f"[{profile}/{split}] missing columns: {sorted(missing)}")

        # ---- format + shuffle -----------------------------------------------
        def _fmt(example):
            return to_prompt_completion(example, tokenizer)

        dataset["train"] = (
            dataset["train"]
            .map(_fmt, remove_columns=dataset["train"].column_names, load_from_cache_file=False)
            .shuffle(seed=SEED)
        )
        dataset["validation"] = dataset["validation"].map(
            _fmt, remove_columns=dataset["validation"].column_names, load_from_cache_file=False
        )

        print(f"[{profile}] train={len(dataset['train'])}  val={len(dataset['validation'])}")
        print(f"[{profile}] sample prompt:\n{dataset['train'][0]['prompt'][:300]}\n")

        # ---- model (full weights, no LoRA) ----------------------------------
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME_OR_PATH,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map={"": 0},
        )

        # ---- training config ------------------------------------------------
        sft_config = SFTConfig(
            output_dir=out_dir,
            run_name=f"sft_{profile}",
            num_train_epochs=EPOCHS,
            # Full-parameter training with gradient accumulation
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
            # Paged AdamW keeps optimiser states in CPU memory pages,
            # essential for fitting a full 8B model on a single GPU.
            optim="paged_adamw_8bit",
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            max_grad_norm=1.0,
            max_length=MAX_LENGTH,
            warmup_ratio=WARMUP_RATIO,
            logging_steps=LOGGING_STEPS,
            save_strategy="epoch",
            save_total_limit=SAVE_TOTAL_LIMIT,
            save_only_model=True,
            eval_strategy="epoch",
            bf16=True,
            report_to=REPORT_TO,
            packing=False,
            completion_only_loss=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            seed=SEED,
            data_seed=SEED,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=tokenizer,
        )

        trainer.train()
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)

        del trainer, model, dataset
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"[{profile}] saved → {out_dir}\n")

    print("All done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)