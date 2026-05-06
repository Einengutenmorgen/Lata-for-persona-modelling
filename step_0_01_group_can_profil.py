"""
build_profiles.py
-----------------
Compose canonical Big Five personality profiles from big5_chat CSV splits.

Canonical profiles (Robins et al. 1996; Donnellan & Robins 2010):
  Resilient     : O↑ C↑ E↑ A↑ N↓
  Overcontrolled: O↓ C≈ E↓ A≈ N↑
  Undercontrolled: O≈ C↓ E↑ A↓ N↑

Level values:
  "high"    → include only high-level rows for that trait
  "low"     → include only low-level rows for that trait
  "neutral" → include an equal 50/50 mix of high and low rows for that trait
               (models a centred / average trait level)

Usage:
  python build_profiles.py \
      --data_root /path/to/DATA_ROOT \
      --out_root  /path/to/OUT_ROOT
"""

import argparse
import csv
import json
import os
import random
import sys


# ---------------------------------------------------------------------------
# Profile definitions
# (trait, level) where level is "high" | "low" | "neutral"
# ---------------------------------------------------------------------------
PROFILES = {
    "Resilient": [
        ("openness",          "high"),
        ("conscientiousness", "high"),
        ("extraversion",      "high"),
        ("agreeableness",     "high"),
        ("neuroticism",       "low"),
    ],
    "Overcontrolled": [
        ("openness",          "low"),
        ("conscientiousness", "neutral"),   # ≈ centred
        ("extraversion",      "low"),
        ("agreeableness",     "neutral"),   # ≈ centred
        ("neuroticism",       "high"),
    ],
    "Undercontrolled": [
        ("openness",          "neutral"),   # ≈ centred
        ("conscientiousness", "low"),
        ("extraversion",      "high"),
        ("agreeableness",     "low"),
        ("neuroticism",       "high"),
    ],
}

SPLITS = ["train", "dev", "test"]

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict]:
    """Read a CSV and return rows as dicts. Fails fast if file is missing."""
    if not os.path.isfile(path):
        sys.exit(f"ERROR: file not found: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"ERROR: {path} is empty")
    required = {"prompt", "answer", "trait", "level"}
    missing = required - set(rows[0].keys())
    if missing:
        sys.exit(f"ERROR: {path} is missing columns: {missing}")
    return rows


def collect_rows_for_trait(
    rows: list[dict],
    trait: str,
    level: str,
    rng: random.Random,
) -> list[dict]:
    """
    Return rows matching (trait, level).

    For level == "neutral": take all high rows and all low rows for this
    trait, then downsample the larger side to match the smaller so the
    final mix is exactly 50 / 50.  A deterministic RNG seed is used so
    results are reproducible.
    """
    by_level = {
        "high": [r for r in rows if _match(r, trait, "high")],
        "low":  [r for r in rows if _match(r, trait, "low")],
    }

    if level in ("high", "low"):
        return by_level[level]

    if level == "neutral":
        high_rows = by_level["high"]
        low_rows  = by_level["low"]
        n = min(len(high_rows), len(low_rows))
        if n == 0:
            sys.exit(f"ERROR: no rows found for trait='{trait}' at either level")
        sample_high = rng.sample(high_rows, n)
        sample_low  = rng.sample(low_rows,  n)
        mixed = sample_high + sample_low
        rng.shuffle(mixed)
        return mixed

    sys.exit(f"ERROR: unknown level '{level}' — must be high | low | neutral")


def _match(row: dict, trait: str, level: str) -> bool:
    return (
        row["trait"].strip().lower() == trait
        and row["level"].strip().lower() == level
    )


def build_profile_rows(
    rows: list[dict],
    conditions: list[tuple],
    rng: random.Random,
) -> list[dict]:
    """Collect and concatenate rows for all (trait, level) pairs in a profile."""
    result = []
    for trait, level in conditions:
        matched = collect_rows_for_trait(rows, trait, level, rng)
        result.extend(matched)
    return result


def write_jsonl(rows: list[dict], path: str) -> None:
    """Write rows as JSONL, one JSON object per line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            record = {
                "prompt": row["prompt"].strip(),
                "answer": row["answer"].strip(),
                "trait":  row["trait"].strip().lower(),
                "level":  row["level"].strip().lower(),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):>6,} rows → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build canonical Big Five profile splits.")
    parser.add_argument("--data_root", required=True,
                        help="Root dir containing big5/big5_{train,dev,test}.csv")
    parser.add_argument("--out_root", required=True,
                        help="Output root; sub-dirs per profile will be created.")
    args = parser.parse_args()

    rng = random.Random(RANDOM_SEED)

    # Load all splits up front — fail fast if any is missing
    split_data = {}
    for split in SPLITS:
        path = os.path.join(args.data_root, "big5", f"big5_{split}.csv")
        split_data[split] = load_csv(path)
        print(f"Loaded {len(split_data[split]):>7,} rows from {path}")

    print()

    for profile_name, conditions in PROFILES.items():
        print(f"[{profile_name}]")
        for trait, level in conditions:
            print(f"  {trait:<20} → {level}")
        for split in SPLITS:
            rows = build_profile_rows(split_data[split], conditions, rng)
            out_path = os.path.join(args.out_root, profile_name, f"{split}.jsonl")
            write_jsonl(rows, out_path)
        print()

    print("Done.")


if __name__ == "__main__":
    main()