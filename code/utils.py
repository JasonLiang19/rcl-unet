import pandas as pd


def process_non_serpins():

    input_file = "../data/non serpins.tsv"

    # Load the full UniProt TSV
    df = pd.read_csv(input_file, sep='\t')
    df.rename(columns={'Entry': 'id'}, inplace=True)

    # Check number of entries
    print(f"Total entries loaded: {len(df)}")

    # Filter out sequences containing 'X'
    filtered_df = df[~df['Sequence'].str.contains('X', na=False)]
    print(f"Remaining entries after removing sequences with 'X': {len(filtered_df)}")

    # randomly split remaining samples
    train_df = df.sample(frac=.5)
    test_df = df.drop(train_df.index)

    # Randomly sample from each (set random_state for reproducibility)
    train_sample = train_df.sample(n=1300, random_state=42)
    test_sample = test_df.sample(n=1024, random_state=42)

    # Save to new TSV
    train_sample.to_csv("../data/non_serpin_train_1300.csv", index=False)
    test_sample.to_csv("../data/non_serpin_test.csv", index=False)

def csv_to_fasta(input_file, output_file, just_rcl: bool):

    input_file = input_file

    df = pd.read_csv(input_file)
    df = df.dropna(subset=['rcl_seq']) # get rid of rows without annotation 

    with open(output_file, 'w') as f:
        for _, row in df.iterrows():
            id = row['id']
            rcl_start = row['rcl_start']
            rcl_end = row['rcl_end']
            if just_rcl:
                sequence = row['rcl_seq']
            else:
                sequence = row['Sequence']
            f.write(f">{id} rcl:{rcl_start}-{rcl_end}\n")
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i+80] + "\n")

def main():
    csv_to_fasta("../data/Alphafold RCL annotations.csv", "../data/RCL ONLY Alphafold annotations.fasta", True)

if __name__ == '__main__':
    main()
