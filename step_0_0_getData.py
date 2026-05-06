#!/usr/bin/env python3

import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def main():
    print("1. Downloading dataset...")
    dataset = load_dataset("wenkai-li/big5_chat")

    # Combine all existing splits into one master DataFrame
    df_all = pd.concat([dataset[split].to_pandas() for split in dataset.keys()], ignore_index=True)

    print("2. Filtering for Big 5 entries...")
    # Make sure we catch traits regardless of capitalization
    valid_traits = ['o', 'c', 'e', 'a', 'n', 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    
    # Convert the trait column to lowercase temporarily just for the filter check
    df_big5 = df_all[df_all['trait'].str.lower().isin(valid_traits)].copy()

    print("3. Finding inputs that appear exactly 10 times...")
    input_counts = df_big5['train_input'].value_counts()
    perfect_10_inputs = input_counts[input_counts == 10].index
    df_filtered = df_big5[df_big5['train_input'].isin(perfect_10_inputs)].copy()

    print("4. Formatting to required columns (prompt, answer, trait, level)...")
    df_ready = df_filtered[['train_input', 'train_output', 'trait', 'level']].rename(columns={
        'train_input': 'prompt',
        'train_output': 'answer'
    })

    print("5. Splitting into Train / Dev / Test...")
    # FIX: Wrap in list() to cast away the PyArrow ExtensionArray so sklearn can slice it
    unique_prompts = list(df_ready['prompt'].unique())
    
    # Quick sanity check
    if len(unique_prompts) == 0:
        print("ERROR: No prompts found after filtering! Exiting.")
        return

    # Now train_test_split will handle the standard Python list without crashing
    train_prompts, temp_prompts = train_test_split(unique_prompts, test_size=0.2, random_state=42)
    dev_prompts, test_prompts = train_test_split(temp_prompts, test_size=0.5, random_state=42)

    df_train = df_ready[df_ready['prompt'].isin(train_prompts)]
    df_dev   = df_ready[df_ready['prompt'].isin(dev_prompts)]
    df_test  = df_ready[df_ready['prompt'].isin(test_prompts)]

    print("6. Saving to CSV...")
    task_name = "big5" 
    data_root = "./DATA_ROOT" 
    task_dir = os.path.join(data_root, task_name)
    os.makedirs(task_dir, exist_ok=True)

    df_train.to_csv(os.path.join(task_dir, f"{task_name}_train.csv"), index=False)
    df_dev.to_csv(os.path.join(task_dir, f"{task_name}_dev.csv"), index=False)
    df_test.to_csv(os.path.join(task_dir, f"{task_name}_test.csv"), index=False)

    print("\n--- Summary ---")
    print(f"Total valid unique prompts: {len(unique_prompts)}")
    print(f"Total extracted rows: {len(df_ready)}")
    print(f"Train rows: {len(df_train)}")
    print(f"Dev rows:   {len(df_dev)}")
    print(f"Test rows:  {len(df_test)}")
    print(f"Files saved in: {task_dir}")

if __name__ == "__main__":
    main()