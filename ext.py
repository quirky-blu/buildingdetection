import pandas as pd

file_path = 'data.csv'
df = pd.read_csv(file_path)

# Shuffle rows for random assignment
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Compute counts
n = len(df)
red_n = int(0.10 * n)
yellow_n = int(0.20 * n)
green_n = n - red_n - yellow_n

# Assign confidence
df.loc[:red_n-1, 'confidence'] = 0.0
df.loc[red_n:red_n+yellow_n-1, 'confidence'] = 0.5
df.loc[red_n+yellow_n:, 'confidence'] = 1.0

# Assign color column directly
df['color'] = df['confidence'].map({1.0: 'green', 0.5: 'yellow', 0.0: 'red'})

# Save new CSV
df.to_csv('data_modified.csv', index=False)
