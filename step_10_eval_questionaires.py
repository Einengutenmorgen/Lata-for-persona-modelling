import argparse

import pandas as pd

SCORING_KEYS = {
    "Extraversion":      [1, -2, 3, -4, 5, -6, 7, -8, 9, -10],
    "Neuroticism":       [11, -12, 13, -14, 15, 16, 17, 18, 19, 20],
    "Agreeableness":     [-21, 22, -23, 24, -25, 26, -27, 28, 29, 30],
    "Conscientiousness": [31, -32, 33, -34, 35, -36, 37, -38, 39, 40],
    "Openness":          [41, -42, 43, -44, 45, -46, 47, 48, 49, 50],
}


def score(df: pd.DataFrame) -> pd.DataFrame:
    results = df[["model"]].copy()
    for trait, items in SCORING_KEYS.items():
        cols = []
        for item in items:
            col = f"X_{abs(item)}"
            if item < 0:
                rev_col = f"{col}_rev"
                df[rev_col] = 6 - df[col]
                cols.append(rev_col)
            else:
                cols.append(col)
        results[trait] = df[cols].mean(axis=1)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="Wide-format responses CSV from step_9 ({out}.responses.csv)")
    ap.add_argument("--out", required=True, help="Output profiles CSV path")
    args = ap.parse_args()

    df = pd.read_csv(args.responses)
    results = score(df)
    profiles = results.groupby("model").mean()
    print(profiles.to_string())
    profiles.to_csv(args.out)
    print(f"\n[ok] wrote: {args.out}")


if __name__ == "__main__":
    main()
