# finetune_personality_sft.py
# Hardcoded configuration version (no argparse), LoRA + SFT training.
#
# Expects per-task:
#   {PROCESSED_ROOT}/{task}/train.jsonl
#   {PROCESSED_ROOT}/{task}/dev.jsonl
#
# Saves adapters to:
#   {OUTPUT_ROOT}/{task}

import gc
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

# =========================
# HARD-CODED CONFIG
# =========================

MODEL_NAME_OR_PATH = "meta-llama/Llama-3.1-8B-Instruct"     # base model
PROCESSED_ROOT = "Task_III/"          # output of prepare_personality_data.py
OUTPUT_ROOT = "models_llama31_8b_sft"             # where adapters will be saved
wandb_id= 'llama_31_8b'
TASKS = (
    "agreeableness_high",
    "agreeableness_low",
   
)

# Training hyperparameters (minimal set)
SEED = 42
EPOCHS = 3.0
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MAX_LENGTH = 512
WARMUP_RATIO = 0.03
LOGGING_STEPS = 100
SAVE_TOTAL_LIMIT = 2

# LoRA hyperparameters
LORA_R = 8
LORA_ALPHA = 16
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Logging
REPORT_TO = "wandb"  # "wandb" or "none"


def require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def fail_fast_env_checks(tokenizer: AutoTokenizer) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; expected a GPU environment.")
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer has no eos_token.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def main() -> None:
    set_seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    fail_fast_env_checks(tokenizer)

    for task in TASKS:
        train_path = os.path.join(PROCESSED_ROOT, task, "train.jsonl")
        dev_path = os.path.join(PROCESSED_ROOT, task, "dev.jsonl")
        require_file(train_path)
        require_file(dev_path)

        out_dir = os.path.join(OUTPUT_ROOT, task)
        os.makedirs(out_dir, exist_ok=True)

        if REPORT_TO == "wandb":
            os.environ["WANDB_NAME"] = f"sft_{wandb_id}_{task}"
            os.environ["WANDB_TAGS"] = f"sft_{wandb_id}_{task}"

        dataset = load_dataset("json", data_files={"train": train_path, "validation": dev_path},
                               keep_in_memory=False)

        # Fail fast on required columns
        needed_cols = {"story", "chosen"}  # <-- changed
        for split in ("train", "validation"):
            cols = set(dataset[split].column_names)
            missing = needed_cols - cols
            if missing:
                raise ValueError(f"[{task}] missing columns in {split}: {sorted(missing)}")


        # def to_prompt_completion(example):
        #     prompt = example["prompt"].rstrip() + "\n"
        #     completion = example["chosen"].rstrip() + tokenizer.eos_token
        #     return {"prompt": prompt, "completion": completion}

        # dataset["train"] = dataset["train"].map(to_prompt_completion,
        #                                         remove_columns=dataset["train"].column_names, 
        #                                         num_proc=1, 
        #                                         load_from_cache_file=False )
        
        # dataset["validation"] = dataset["validation"].map(
        #     to_prompt_completion, remove_columns=dataset["validation"].column_names
        # )

        SYSTEM_MSG = "You are a helpful assistant."

        def to_chat_text(example):
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": example["story"].rstrip()},
                {"role": "assistant", "content": example["chosen"].rstrip()},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
            return {"text": text}

        # APPLY MAPPING 
        dataset["train"] = dataset["train"].map(
            to_chat_text,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
        )
        dataset["validation"] = dataset["validation"].map(
            to_chat_text,
            remove_columns=dataset["validation"].column_names,
            load_from_cache_file=False,
        )
        print(f"[{task}] sample text:\n{dataset['train'][0]['text'][:400]}\n")


        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME_OR_PATH,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,  
            device_map={"": 0},
        )

        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        )

        sft_config = SFTConfig(
            output_dir=out_dir,
            run_name=f"sft_{task}",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            max_length=MAX_LENGTH,
            warmup_ratio=WARMUP_RATIO,
            logging_steps=LOGGING_STEPS,
            save_strategy="epoch",
            save_total_limit=SAVE_TOTAL_LIMIT,
            eval_strategy="epoch",
            bf16=True,
            report_to=REPORT_TO,
            packing=False,
            dataset_text_field="text",
            completion_only_loss=False,
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
            peft_config=peft_config,
            processing_class=tokenizer,
        )

        trainer.train()
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)

        del trainer, model, dataset
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'synchronize'):
            torch.cuda.synchronize()

        print(f"[{task}] saved -> {out_dir}")

    print("All done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
