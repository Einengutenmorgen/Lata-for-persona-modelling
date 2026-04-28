import re

from transformers import AutoTokenizer

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


def layer_id(name: str) -> int:
    m = LAYER_RE.match(name)
    return int(m.group(1)) if m else -1


def is_target_param(name: str) -> bool:
    return name.startswith("model.layers.") and any(s in name for s in TARGET_SUBSTRINGS)


def load_tokenizer(path: str) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(path, use_fast=True, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(path, use_fast=True)
