#step_8_combine.py
import pandas as pd

dfs = []
for path in [
    "artifacts/agreeableness_high_linear_lambda_sweep.csv",
    "artifacts/agreeableness_high_log_lambda_sweep.csv",
    "artifacts/agreeableness_high_thr2e-4_lambda_sweep.csv",
]:
    df = pd.read_csv(path)
    df["source_csv"] = path
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# Keep only the sweeps (exclude baselines duplicates if you want)
print(all_df.sort_values(["run", "lambda"], na_position="first"))
