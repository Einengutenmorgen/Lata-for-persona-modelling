#step_3_weights.py
import argparse
import json
import math
from pathlib import Path

import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pt", required=True, help="path to layer_cosine.pt (torch.save output)")
    ap.add_argument("--out_json", required=True, help="output json for layer weights")
    ap.add_argument("--method", required=True, choices=["linear", "log", "threshold"])
    ap.add_argument("--sigma", type=float, default=0.95, help="threshold for method=threshold")
    args = ap.parse_args()

    obj = torch.load(args.in_pt, map_location="cpu")
    cos = obj.get("layer_cosine", None)
    if cos is None or not isinstance(cos, dict) or len(cos) == 0:
        raise RuntimeError("Input .pt does not contain a non-empty 'layer_cosine' dict.")

    # normalize keys to int
    cos = {int(k): float(v) for k, v in cos.items()}
    layers = sorted(cos.keys())
    L = max(layers) + 1

    # fail fast if missing layers (expected 0..L-1)
    missing = [i for i in range(L) if i not in cos]
    if missing:
        raise RuntimeError(f"Missing cosine entries for layers: {missing}")

    weights = {}
    ranks = {}

    if args.method in ["linear", "log"]:
        # Rank by cosine similarity (highest cosine = rank 1; lowest cosine = rank L)
        ranked = sorted(cos.items(), key=lambda kv: kv[1], reverse=True)
        for idx, (layer, _) in enumerate(ranked):
            r = idx + 1  # 1..L
            ranks[layer] = r
            if args.method == "linear":
                w = r / L
            else:
                # log base L: log_L(r). rank=1 -> 0, rank=L -> 1
                w = math.log(r) / math.log(L)
            weights[layer] = float(w)

    elif args.method == "threshold":
        for layer, c in cos.items():
            weights[layer] = 0.0 if c >= args.sigma else 1.0

    out = {
        "source_pt": str(args.in_pt),
        "method": args.method,
        "sigma": args.sigma if args.method == "threshold" else None,
        "L": L,
        "layer_cosine": cos,
        "layer_rank": ranks if ranks else None,
        "layer_weight": weights,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[ok] wrote: {out_path}")

    # small summary
    w_items = sorted(weights.items(), key=lambda kv: kv[1])
    print("[weights] smallest 5:", w_items[:5])
    print("[weights] largest  5:", w_items[-5:])

if __name__ == "__main__":
    main()
