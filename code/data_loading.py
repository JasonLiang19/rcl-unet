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

# define problem properties
FASTA_RESIDUE_LIST = ["A", "D", "N", "R", "C", "E", "Q", "G", "H", "I",
                      "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
NB_RESIDUES = len(FASTA_RESIDUE_LIST)
RESIDUE_DICT = dict(zip(FASTA_RESIDUE_LIST, range(NB_RESIDUES)))
UPPER_LENGTH_LIMIT = 1024


def read_fasta(filepath: str):
    # Read all non-empty lines from FASTA
    with open(filepath, 'r') as reader:
        lines = [line.strip() for line in reader if line.strip() != '']
    
    protein_names = []
    sequences = []
    new_sequence = True
    for line in lines:
        if line.startswith((">", ";")):
            protein_names.append(line[1:].strip())
            new_sequence = True
        elif new_sequence:
            sequences.append(line)
            new_sequence = False
        else:
            sequences[-1] = f"{sequences[-1]}{line}"
    
    data_dict = defaultdict(dict)
    for protein_name, resnames in zip(protein_names, sequences):
        sequence = to_categorical([RESIDUE_DICT[residue] for residue in resnames], num_classes=NB_RESIDUES)
        data_dict[protein_name]["fasta"] = sequence

    print(len(data_dict), "proteins loaded")

    return data_dict

def read_train_csv(filepath: str):

    data_dict = defaultdict(dict)

    # Load entire CSV as DataFrame
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['rcl_seq']) # get rid of rows without annotation 
    df = df[df["Sequence"].str.len() <= UPPER_LENGTH_LIMIT]

    print("Loading ProtTrans model")   
    embedder = OfflineProtTransT5XLU50Embedder()
    
    # Iterate row by row
    for _, row in df.iterrows():

        protein_id = row["id"].strip()

        # fasta
        sequence = row["Sequence"].strip()
        data_dict[protein_id]["sequence"] = sequence

        # one-hot encoded sequence 
        encoded_sequence = to_categorical([RESIDUE_DICT[residue] for residue in sequence], num_classes=NB_RESIDUES)
        data_dict[protein_id]["one-hot"] = encoded_sequence

        # label 
        rcl_start = int(row['rcl_start'])
        rcl_end = int(row['rcl_end'])

        seq_len = len(sequence)
        rcl_label = np.full((UPPER_LENGTH_LIMIT, 2), [1, 0], dtype=np.float32)  # all non-RCL by default

        # Apply RCL labels (convert to 0-based indexing)
        rcl_start_idx = max(0, rcl_start - 1)
        rcl_end_idx = min(seq_len, rcl_end)  # do not exceed actual length

        for i in range(rcl_start_idx, rcl_end_idx):
            rcl_label[i] = [0, 1]

        # Mask out padding if sequence is shorter than max_length
        for i in range(seq_len, UPPER_LENGTH_LIMIT):
            rcl_label[i] = [9999, 9999]

        data_dict[protein_id]['label'] = rcl_label

    # prottrans    
    for protein_name in tqdm(data_dict, desc='Calculating ProtTrans Features'):
        # uses unencoded sequence
        data_dict[protein_name]["prottrans"] = embedder.embed(data_dict[protein_name]["sequence"])
    
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
            print('a')
            # model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
            model = T5EncoderModel.from_pretrained(self._model_directory)
        else:
            print('b')
            model = T5Model.from_pretrained(self._model_directory)
        return model