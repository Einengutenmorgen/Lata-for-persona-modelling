#!/usr/bin/env python3
"""
Step 11: Simplified Personality Text Generation Evaluation

A standalone script that evaluates models on personality-revealing text generation
using prompts inspired by PsychAdapter methodology. 

Supports both:
- Chat format (for Instruct/SFT models trained with chat template)
- Completion format (for base models without chat template)

Evaluates generated text using KevSun/Personality_LM for Big Five profiling.

Usage:
    # Evaluate a single model with chat template
    python step_11_OEG_personality_text_gen_simple.py \
        --model_path meta-llama/Llama-3.1-8B-Instruct \
        --model_name instruct_baseline \
        --use_chat_template \
        --output_dir ./results

    # Evaluate a base model without chat template
    python step_11_OEG_personality_text_gen_simple.py \
        --model_path meta-llama/Llama-3.1-8B \
        --model_name base \
        --output_dir ./results

    # Evaluate multiple models from a config file
    python step_11_OEG_personality_text_gen_simple.py \
        --config step_11_OEG_models_config_example.json \
        --output_dir ./results
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration - MUST MATCH YOUR TRAINING CONFIG
# =============================================================================

# System message used during training (from step_0_2_sft_seq.py)
SYSTEM_MSG = "Answer in person!"

# =============================================================================
# Prompts inspired by PsychAdapter for personality-relevant text generation
# =============================================================================

# Chat-formatted prompts (for instruct models)
CHAT_PROMPTS = {
    "i_like_to": "Complete this sentence naturally: I like to",
    "i_hate_when": "Complete this sentence naturally: I hate when",
    "my_hobbies_are": "Complete this sentence naturally: My hobbies are",
    "in_my_free_time": "Complete this sentence naturally: In my free time",
    "i_feel": "Complete this sentence naturally: I feel",
    "at_parties": "Complete this sentence naturally: At parties, I",
    "i_worry_about": "Complete this sentence naturally: I worry about",
    "my_goals": "Complete this sentence naturally: My goals are",
    "helping_others": "Complete this sentence naturally: Helping others",
    "new_experiences": "Complete this sentence naturally: New experiences",
    "when_stressed": "Complete this sentence naturally: When I'm stressed, I",
    "relationships": "Complete this sentence naturally: Relationships",
}

# Raw prompts (for base models / completion mode)
RAW_PROMPTS = {
    "i_like_to": "I like to",
    "i_hate_when": "I hate when",
    "my_hobbies_are": "My hobbies are",
    "in_my_free_time": "In my free time",
    "i_feel": "I feel",
    "at_parties": "At parties, I",
    "i_worry_about": "I worry about",
    "my_goals": "My goals are",
    "helping_others": "Helping others",
    "new_experiences": "New experiences",
    "when_stressed": "When I'm stressed, I",
    "relationships": "Relationships",
}

# Big Five trait names matching KevSun/Personality_LM output order
TRAIT_NAMES = ["agreeableness", "openness", "conscientiousness", "extraversion", "neuroticism"]


# =============================================================================
# Utility Functions
# =============================================================================

def has_chat_template(tokenizer) -> bool:
    """Check if tokenizer has a chat template."""
    return hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None


def build_chat_prompt(tokenizer, user_content: str) -> str:
    """Build prompt using chat template format matching training."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_content.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


# =============================================================================
# Personality Evaluator
# =============================================================================

class PersonalityLM:
    """Wrapper for KevSun/Personality_LM personality detection model."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        print("[PersonalityLM] Loading KevSun/Personality_LM...")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "KevSun/Personality_LM",
            ignore_mismatched_sizes=True,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("KevSun/Personality_LM")
        self.model.eval()
        print("[PersonalityLM] Model loaded successfully")
    
    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, float]:
        """Predict Big Five personality scores for text."""
        if not text.strip():
            return {t: 0.0 for t in TRAIT_NAMES}
        
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        outputs = self.model(**encoded)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        scores = probs[0].cpu().tolist()
        
        return dict(zip(TRAIT_NAMES, scores))
    
    def compute_mean_profile(self, texts: List[str]) -> Dict[str, float]:
        """Compute mean personality profile from multiple texts."""
        all_scores = {t: [] for t in TRAIT_NAMES}
        
        for text in texts:
            if not text.strip():
                continue
            scores = self.predict(text)
            for trait, score in scores.items():
                all_scores[trait].append(score)
        
        return {
            trait: sum(scores) / len(scores) if scores else 0.0
            for trait, scores in all_scores.items()
        }


# =============================================================================
# Text Generator
# =============================================================================

class TextGenerator:
    """Handles text generation from language models."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
        # Ensure padding token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @torch.no_grad()
    def generate(
        self, 
        prompt: str, 
        use_chat_template: bool = True,
        seed: Optional[int] = None,
        debug: bool = False,
    ) -> str:
        """Generate text completion for a prompt."""
        if seed is not None:
            torch.manual_seed(seed)
        
        # Build the actual prompt string
        if use_chat_template and has_chat_template(self.tokenizer):
            prompt_str = build_chat_prompt(self.tokenizer, prompt)
        else:
            prompt_str = prompt
        
        if debug:
            print(f"[DEBUG] Use chat template: {use_chat_template and has_chat_template(self.tokenizer)}")
            print(f"[DEBUG] Prompt:\n{prompt_str[:500]}...")
        
        inputs = self.tokenizer(prompt_str, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Extract only generated portion
        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = outputs[0][prompt_len:]
        generated = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        
        if debug:
            print(f"[DEBUG] Generated: {generated[:200]}...")
        
        return generated
    
    def generate_for_prompts(
        self,
        prompts: Dict[str, str],
        use_chat_template: bool = True,
        num_samples: int = 5,
        seed_base: int = 42,
        debug: bool = False,
    ) -> Dict[str, List[str]]:
        """Generate multiple samples for each prompt."""
        results = {}
        
        for i, (name, prompt) in enumerate(tqdm(prompts.items(), desc="Generating")):
            samples = []
            for j in range(num_samples):
                seed = seed_base + j * 1000
                text = self.generate(
                    prompt, 
                    use_chat_template=use_chat_template,
                    seed=seed,
                    debug=(debug and i == 0 and j == 0)
                )
                samples.append(text)
            results[name] = samples
        
        return results


# =============================================================================
# Main Evaluation
# =============================================================================

def load_model(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """Load a model and tokenizer."""
    print(f"[load] Loading model: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": 0} if device == "cuda" else "cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    return model, tokenizer


def evaluate_model(
    model_path: str,
    model_name: str,
    personality_lm: PersonalityLM,
    use_chat_template: bool = True,
    num_samples: int = 5,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    seed: int = 42,
    device: str = "cuda",
    debug: bool = False,
) -> tuple:
    """Evaluate a single model and return results."""
    
    # Load model
    model, tokenizer = load_model(model_path, device)
    
    # Create generator
    generator = TextGenerator(
        model, tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    
    # Choose prompts based on chat template usage
    prompts = CHAT_PROMPTS if use_chat_template else RAW_PROMPTS
    
    # Generate texts
    generations = generator.generate_for_prompts(
        prompts, 
        use_chat_template=use_chat_template,
        num_samples=num_samples, 
        seed_base=seed,
        debug=debug,
    )
    
    # Collect all generated texts
    all_texts = []
    generation_records = []
    
    for prompt_name, samples in generations.items():
        raw_prompt = RAW_PROMPTS[prompt_name]  # Use raw for personality context
        for i, text in enumerate(samples):
            full_text = f"{raw_prompt} {text}"
            all_texts.append(full_text)
            
            # Get personality score for this text
            scores = personality_lm.predict(full_text)
            
            record = {
                "model_name": model_name,
                "model_path": model_path,
                "use_chat_template": use_chat_template,
                "prompt_name": prompt_name,
                "prompt": raw_prompt,
                "sample_idx": i,
                "generated_text": text,
                "full_text": full_text,
            }
            record.update(scores)
            generation_records.append(record)
    
    # Compute mean profile
    profile = personality_lm.compute_mean_profile(all_texts)
    profile["model_name"] = model_name
    profile["model_path"] = model_path
    profile["use_chat_template"] = use_chat_template
    profile["n_samples"] = len(all_texts)
    
    # Cleanup
    del model, generator
    torch.cuda.empty_cache()
    
    return profile, generation_records


def main():
    parser = argparse.ArgumentParser(
        description="Personality text generation evaluation"
    )
    
    # Model specification (choose one)
    parser.add_argument("--model_path", type=str,
                        help="Path to a single model")
    parser.add_argument("--model_name", type=str, default="model",
                        help="Name for the model (used in output)")
    parser.add_argument("--config", type=str,
                        help="Path to JSON config file with multiple models")
    
    # Chat template option
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Use chat template format (for instruct/SFT models)")
    parser.add_argument("--no_chat_template", action="store_true",
                        help="Do not use chat template (for base models)")
    
    # Generation settings
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples per prompt")
    parser.add_argument("--seed", type=int, default=42)
    
    # Prompt selection
    parser.add_argument("--prompts", type=str, default="",
                        help="Comma-separated prompt names (empty=all)")
    
    # Output
    parser.add_argument("--output_dir", default="./personality_results")
    
    # Hardware
    parser.add_argument("--device", default="cuda")
    
    # Debug
    parser.add_argument("--debug", action="store_true",
                        help="Print debug information")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_path and not args.config:
        parser.error("Must specify either --model_path or --config")
    
    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Filter prompts if specified
    if args.prompts:
        selected = args.prompts.split(",")
        global CHAT_PROMPTS, RAW_PROMPTS
        CHAT_PROMPTS = {k: v for k, v in CHAT_PROMPTS.items() if k in selected}
        RAW_PROMPTS = {k: v for k, v in RAW_PROMPTS.items() if k in selected}
    
    print(f"Using {len(CHAT_PROMPTS)} prompts: {list(CHAT_PROMPTS.keys())}")
    
    # Load models to evaluate
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        models_config = config.get("models", config)  # Support both formats
    else:
        # Determine chat template usage
        use_chat = args.use_chat_template or (not args.no_chat_template and "instruct" in args.model_path.lower())
        models_config = [{
            "name": args.model_name, 
            "path": args.model_path,
            "use_chat_template": use_chat,
        }]
    
    # Initialize personality evaluator
    personality_lm = PersonalityLM(device=args.device)
    
    # Evaluate each model
    all_profiles = []
    all_generations = []
    
    for model_cfg in models_config:
        name = model_cfg.get("name", "model")
        path = model_cfg.get("path")
        
        # Determine chat template usage from config or heuristic
        use_chat = model_cfg.get("use_chat_template")
        if use_chat is None:
            # Heuristic: use chat template for instruct/sft models
            model_type = model_cfg.get("type", "")
            use_chat = model_type in ["instruct", "sft"] or "instruct" in path.lower()
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"  Path: {path}")
        print(f"  Use chat template: {use_chat}")
        print(f"{'='*60}")
        
        try:
            profile, generations = evaluate_model(
                model_path=path,
                model_name=name,
                personality_lm=personality_lm,
                use_chat_template=use_chat,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed,
                device=args.device,
                debug=args.debug,
            )
            
            all_profiles.append(profile)
            all_generations.extend(generations)
            
            # Print profile
            print(f"\nPersonality Profile for {name}:")
            for trait in TRAIT_NAMES:
                print(f"  {trait}: {profile[trait]:.4f}")
                
        except Exception as e:
            print(f"[error] Failed to evaluate {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if all_profiles:
        profiles_df = pd.DataFrame(all_profiles)
        profiles_path = output_dir / f"profiles_{timestamp}.csv"
        profiles_df.to_csv(profiles_path, index=False)
        print(f"\nSaved profiles to: {profiles_path}")
    
    if all_generations:
        generations_df = pd.DataFrame(all_generations)
        generations_path = output_dir / f"generations_{timestamp}.csv"
        generations_df.to_csv(generations_path, index=False)
        print(f"Saved generations to: {generations_path}")
    
    # Print summary
    if all_profiles:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        summary_df = pd.DataFrame(all_profiles).set_index("model_name")[TRAIT_NAMES]
        print(summary_df.to_string())


if __name__ == "__main__":
    main()