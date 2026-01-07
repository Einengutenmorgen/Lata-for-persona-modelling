import pandas as pd

# Load your questionnaire data
df = pd.read_csv('seq.responses.csv')

# Scoring Keys for IPIP-50 (Numbers refer to X_n, negative means reverse-scored)
keys = {
        'Extraversion': [1, -2, 3, -4, 5, -6, 7, -8, 9, -10],
        'Neuroticism': [11, -12, 13, -14, 15, 16, 17, 18, 19, 20],
        'Agreeableness': [-21, 22, -23, 24, -25, 26, -27, 28, 29, 30],
        'Conscientiousness': [31, -32, 33, -34, 35, -36, 37, -38, 39, 40],
        'Openness': [41, -42, 43, -44, 45, -46, 47, 48, 49, 50]
    }

# Calculate Scores
results = df[['model']].copy()
for trait, items in keys.items():
    trait_cols = []
    for item in items:
        col = f'X_{abs(item)}'
        if item < 0:
            # Reverse score on a 1-5 scale: (Max + Min) - Score
            df[f'{col}_rev'] = 6 - df[col]
            trait_cols.append(f'{col}_rev')
        else:
            trait_cols.append(col)
    results[trait] = df[trait_cols].mean(axis=1)

# Aggregate profiles by model
profiles = results.groupby('model').mean()
print(profiles)


profiles.to_csv('seq_profiles.csv')