import pandas as pd

# Replace this with your actual file path
input_file = "../data/non serpins.tsv"

# Load the full UniProt TSV
df = pd.read_csv(input_file, sep='\t')

# Check number of entries
print(f"Total entries loaded: {len(df)}")

# Filter out sequences containing 'X'
filtered_df = df[~df['Sequence'].str.contains('X', na=False)]

print(f"Remaining entries after removing sequences with 'X': {len(filtered_df)}")

# Randomly sample ~1000 entries (set random_state for reproducibility)
sample_size = 1000
sampled_df = filtered_df.sample(n=sample_size, random_state=42)

df.rename(columns={'Entry': 'id'}, inplace=True)

# Save to new TSV
output_file = "../data/non_serpins_1000.csv"
sampled_df.to_csv(output_file, index=False)

print(f"Sampled {sample_size} entries and saved to {output_file}")
