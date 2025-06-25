import os
import csv
from glob import glob
import numpy as np
from tensorflow.keras.utils import to_categorical
from collections import defaultdict
from copy import deepcopy
import pandas as pd
from transformers import T5Model, T5EncoderModel
from bio_embeddings.embed import ProtTransT5XLU50Embedder
from tqdm import tqdm 
import json
import pickle

# define problem properties
FASTA_RESIDUE_LIST = ["A", "D", "N", "R", "C", "E", "Q", "G", "H", "I",
                      "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
NB_RESIDUES = len(FASTA_RESIDUE_LIST)
RESIDUE_DICT = dict(zip(FASTA_RESIDUE_LIST, range(NB_RESIDUES)))
UPPER_LENGTH_LIMIT = 1024


# def read_fasta(filepath: str):
#     # Read all non-empty lines from FASTA
#     with open(filepath, 'r') as reader:
#         lines = [line.strip() for line in reader if line.strip() != '']
    
#     protein_names = []
#     sequences = []
#     new_sequence = True
#     for line in lines:
#         if line.startswith((">", ";")):
#             protein_names.append(line[1:].strip())
#             new_sequence = True
#         elif new_sequence:
#             sequences.append(line)
#             new_sequence = False
#         else:
#             sequences[-1] = f"{sequences[-1]}{line}"
    
#     data_dict = defaultdict(dict)
#     for protein_name, resnames in zip(protein_names, sequences):
#         sequence = to_categorical([RESIDUE_DICT[residue] for residue in resnames], num_classes=NB_RESIDUES)
#         data_dict[protein_name]["fasta"] = sequence

#     print(len(data_dict), "proteins loaded")

#     return data_dict

def read_seq_file(filepath: str, encoding: str):

    # existing pickled data_dict 
    if filepath.endswith('.pkl'):
        print("📂 Loading existing data_dict...")
        with open(filepath, "rb") as f:
            data_dict = pickle.load(f)
            return data_dict

    data_dict = defaultdict(dict)

    # Load entire CSV as DataFrame
    df = pd.read_csv(filepath)
    df = df[df["Sequence"].str.len() <= UPPER_LENGTH_LIMIT]

    if 'rcl_seq' in df.columns:
        df = df.dropna(subset=['rcl_seq']) # get rid of rows without annotation 
    
    # Iterate row by row
    for _, row in df.iterrows():

        protein_id = row["id"].strip()

        # fasta
        sequence = row["Sequence"].strip()
        data_dict[protein_id]["sequence"] = sequence

        # one-hot encoded sequence 
        # encoded_sequence = to_categorical([RESIDUE_DICT[residue] for residue in sequence], num_classes=NB_RESIDUES)
        # data_dict[protein_id]["one-hot"] = encoded_sequence

        # create labels 
        rcl_label = np.full((UPPER_LENGTH_LIMIT, 2), [1, 0], dtype=np.float32)  # all non-RCL by default

        # Apply RCL labels (convert to 0-based indexing)
        if 'rcl_seq' in df.columns:

            rcl_start = int(row['rcl_start'])
            rcl_end = int(row['rcl_end'])
            rcl_start_idx = max(0, rcl_start - 1)
            rcl_end_idx = min(len(sequence), rcl_end)  # do not exceed actual length

            for i in range(rcl_start_idx, rcl_end_idx):
                rcl_label[i] = [0, 1]

        # Mask out padding if sequence is shorter than max_length
        for i in range(len(sequence), UPPER_LENGTH_LIMIT):
            rcl_label[i] = [9999, 9999]

        data_dict[protein_id]['label'] = rcl_label

    # prottrans    
    if (encoding == 'prottrans'):
        print("Loading ProtTrans model")   
        embedder = OfflineProtTransT5XLU50Embedder()
        for protein_name in tqdm(data_dict, desc='Calculating ProtTrans Features'):
            # uses unencoded sequence
            data_dict[protein_name]["encoding"] = embedder.embed(data_dict[protein_name]["sequence"])

    # one-hot
    if (encoding == 'onehot'):
        with open("../data/encodings/One_hot.json") as f:
            encoding_map = json.load(f)
        for protein_name in tqdm(data_dict, desc='Generating One-hot Encodings'):
            sequence = data_dict[protein_name]["sequence"]
            one_hot_encoded = [encoding_map.get(residue, encoding_map["X"]) for residue in sequence]
            data_dict[protein_name]["encoding"] = np.array(one_hot_encoded, dtype=np.float32)

    # BLOSUM
    if (encoding == 'blosum'):
        with open("../data/encodings/BLOSUM62.json") as f:
            encoding_map = json.load(f)
        for protein_name in tqdm(data_dict, desc='Generating BLOSUM62 Encodings'):
            sequence = data_dict[protein_name]["sequence"]
            blosum_encoded = [encoding_map.get(residue, encoding_map["X"]) for residue in sequence]
            data_dict[protein_name]["encoding"] = np.array(blosum_encoded, dtype=np.float32)
    
    return data_dict


def standardize_data(data_dict: dict):

    mean = np.load(os.path.join("data_stats", "train_mean_prottrans.npy"))
    std = np.load(os.path.join("data_stats", "train_std_prottrans.npy"))

    for key in data_dict.keys():
        data_dict[key]["prottrans"] = (data_dict[key]["prottrans"] - mean) / std

    return data_dict


def fill_array_with_value(array: np.array, length_limit: int, value):

    filler = value * np.ones((length_limit - array.shape[0], array.shape[1]), array.dtype)
    filled_array = np.concatenate((array, filler))

    return filled_array


def fill_with_zeros(data: dict, max_sequence_length: int):
    data_copy = deepcopy(data)
    for key, values in data_copy.items():
        if len(values) == UPPER_LENGTH_LIMIT:
            continue
        elif len(values) > UPPER_LENGTH_LIMIT:
            data_copy[key] = values[:UPPER_LENGTH_LIMIT]
            continue
        data_copy[key] = fill_array_with_value(values, max_sequence_length, 0)

    return data_copy

class OfflineProtTransT5XLU50Embedder(ProtTransT5XLU50Embedder):
    # Use an offline model directory
    def __init__(self, **kwargs):
        self.necessary_directories = []
        super().__init__(model_directory=os.path.join('../data/models', "prot_t5_xl_uniref50"))
        self._half_precision_model = False

    def get_model(self):
        if not self._decoder:
            print('Using T5EncoderModel')
            # model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
            model = T5EncoderModel.from_pretrained(self._model_directory)
        else:
            print('Using T5Model')
            model = T5Model.from_pretrained(self._model_directory)
        return model
    
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


