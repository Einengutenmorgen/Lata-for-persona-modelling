# prepare_personality_data.py
# Converts your existing 3-way split CSVs into JSONL without re-splitting.
#
# Expected per task folder:
#   {DATA_ROOT}/{task}/{task}_train.csv
#   {DATA_ROOT}/{task}/{task}_dev.csv
#   {DATA_ROOT}/{task}/{task}_test.csv
#
# Input CSV columns:
#   prompt, answer, trait, level
#
# Output:
#   {OUT_ROOT}/{task}/train.jsonl
#   {OUT_ROOT}/{task}/dev.jsonl
#   {OUT_ROOT}/{task}/test.jsonl
#
# Fails fast on missing files / missing columns / parse failures (in --strict mode).

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Tuple


_STORY_RE = re.compile(r"Story:\s*(?P<q>\"\"|\"?)(?P<story>.*?)(?P=q)\s*Options:", re.DOTALL)
_OPTS_RE = re.compile(
    r"Options:\s*A\.\s*(?P<a>.*?)\s*B\.\s*(?P<b>.*?)(?:\s*Answer\s+only|\s*Answer:)",
    re.DOTALL,
)


def _strip_wrapping_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "“", "”"}:
        return s[1:-1].strip()
    return s


def parse_story_and_options(raw_prompt: str) -> Tuple[str, str, str]:
    m_story = _STORY_RE.search(raw_prompt)
    if not m_story:
        raise ValueError("Could not parse 'Story: ... Options:' section.")
    story = _strip_wrapping_quotes(m_story.group("story").strip())

    m_opts = _OPTS_RE.search(raw_prompt)
    if not m_opts:
        raise ValueError("Could not parse 'Options: A. ... B. ...' section.")
    opt_a = _strip_wrapping_quotes(m_opts.group("a").strip())
    opt_b = _strip_wrapping_quotes(m_opts.group("b").strip())

    if not story or not opt_a or not opt_b:
        raise ValueError("Parsed empty story/option (check formatting).")

    return story, opt_a, opt_b


def build_structured_prompt(story: str, trait: str, level: str) -> str:
    return (
        "Answer to the following input.\n"
        f"Trait: {trait}\n"
        f"Level: {level}\n"
        "Story:\n"
        f"{story}\n\n"
        "Write a single reply that matches the trait and level.\n"
        "Reply:"
    )


def require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def read_and_convert(csv_path: str, strict: bool) -> List[Dict]:
    required = {"prompt", "answer", "trait", "level"}
    rows_out: List[Dict] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing columns {sorted(missing)} in {csv_path}")

        for idx, row in enumerate(reader, start=1):
            ans = row["answer"].strip()
            if ans not in {"A", "B"}:
                msg = f"{os.path.basename(csv_path)} row {idx}: answer must be 'A' or 'B', got '{ans}'"
                if strict:
                    raise ValueError(msg)
                continue

            try:
                story, a, b = parse_story_and_options(row["prompt"])
            except Exception as e:
                msg = f"{os.path.basename(csv_path)} row {idx}: parse failed: {e}"
                if strict:
                    raise ValueError(msg)
                continue

            trait = row["trait"].strip()
            level = row["level"].strip()

            chosen = a if ans == "A" else b
            rejected = b if ans == "A" else a

            rows_out.append(
                {
                    "prompt": build_structured_prompt(story, trait, level),
                    "chosen": chosen,
                    "rejected": rejected,
                    "trait": trait,
                    "level": level,
                    "story": story,
                    "source_answer": ans,
                    "source_row": idx,
                }
            )

    if strict and len(rows_out) == 0:
        raise RuntimeError(f"Strict mode: produced 0 rows from {csv_path}")

    if len(rows_out) < 5:
        raise RuntimeError(f"Too few usable rows ({len(rows_out)}) in {csv_path}; parsing likely broken.")

    return rows_out


def write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True,
                    help="Root containing task folders: {task}/{task}_{train,dev,test}.csv")
    ap.add_argument("--out-root", type=str, required=True, help="Output root for per-task JSONL.")
    ap.add_argument("--tasks", type=str, default="",
                    help="Comma-separated task folder names. If empty, process all subfolders.")
    ap.add_argument("--strict", action="store_true", help="Fail on first bad row / parse error.")
    args = ap.parse_args()

    if not os.path.isdir(args.data_root):
        raise NotADirectoryError(f"--data-root is not a directory: {args.data_root}")

    if args.tasks.strip():
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = sorted([d for d in os.listdir(args.data_root)
                        if os.path.isdir(os.path.join(args.data_root, d))])

    if not tasks:
        raise ValueError("No tasks found to process.")

    for task in tasks:
        task_dir = os.path.join(args.data_root, task)
        train_csv = os.path.join(task_dir, f"{task}_train.csv")
        dev_csv = os.path.join(task_dir, f"{task}_dev.csv")
        test_csv = os.path.join(task_dir, f"{task}_test.csv")

        require_file(train_csv)
        require_file(dev_csv)
        require_file(test_csv)

        train_rows = read_and_convert(train_csv, strict=args.strict)
        dev_rows = read_and_convert(dev_csv, strict=args.strict)
        test_rows = read_and_convert(test_csv, strict=args.strict)

        out_task_dir = os.path.join(args.out_root, task)
        write_jsonl(os.path.join(out_task_dir, "train.jsonl"), train_rows)
        write_jsonl(os.path.join(out_task_dir, "dev.jsonl"), dev_rows)
        write_jsonl(os.path.join(out_task_dir, "test.jsonl"), test_rows)

        print(
            f"[{task}] "
            f"train={len(train_rows)} dev={len(dev_rows)} test={len(test_rows)} -> {out_task_dir}"
        )

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
