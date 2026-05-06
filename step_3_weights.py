# step_3_weights.py
# Converts per-layer cosine similarities (from step_2) into LATA layer weights.
#
# Three methods:
#   linear    : w_l = rank / L          (rank 1 = highest cosine = lowest weight)
#   log       : w_l = log_L(rank)       (same direction, log-compressed)
#   threshold : w_l = 0 if cos >= sigma else 1   (binary mask)

import argparse
import json
import math
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pt",    required=True, help="layer_cosine.pt from step_2")
    ap.add_argument("--out_json", required=True, help="output JSON for layer weights")
    ap.add_argument("--method",   required=True, choices=["linear", "log", "threshold"])
    ap.add_argument("--sigma",    type=float, default=0.95,
                    help="cosine threshold for method=threshold")
    args = ap.parse_args()

    obj = torch.load(args.in_pt, map_location="cpu", weights_only=True)
    cos = obj.get("layer_cosine")
    if not cos or not isinstance(cos, dict):
        raise RuntimeError("Input .pt does not contain a non-empty 'layer_cosine' dict.")

    cos    = {int(k): float(v) for k, v in cos.items()}
    layers = sorted(cos.keys())
    L      = max(layers) + 1

    missing = [i for i in range(L) if i not in cos]
    if missing:
        raise RuntimeError(f"Missing cosine entries for layers: {missing}")

    weights = {}
    ranks   = {}

    if args.method in ("linear", "log"):
        # rank 1 = highest cosine similarity = most similar to instruct vector
        # → gets the LOWEST weight (we want to preserve those layers)
        # Sort descending by cosine, with layer index as tiebreaker for reproducibility.
        ranked = sorted(cos.items(), key=lambda kv: (-kv[1], kv[0]))
        for idx, (layer, _) in enumerate(ranked):
            r = idx + 1      # 1 .. L
            ranks[layer] = r
            if args.method == "linear":
                w = r / L
            else:            # log: log_L(r), maps [1..L] → [0..1]
                w = math.log(r) / math.log(L)
            weights[layer] = float(w)

    elif args.method == "threshold":
        for layer, c in cos.items():
            weights[layer] = 0.0 if c >= args.sigma else 1.0

    out = {
        "source_pt":    str(args.in_pt),
        "method":       args.method,
        "sigma":        args.sigma if args.method == "threshold" else None,
        "L":            L,
        "layer_cosine": cos,
        "layer_rank":   ranks if ranks else None,
        "layer_weight": weights,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[ok] wrote: {out_path}")

    w_items = sorted(weights.items(), key=lambda kv: kv[1])
    print("[weights] smallest 5:", w_items[:5])
    print("[weights] largest  5:", w_items[-5:])


if __name__ == "__main__":
    main()