import os
from glob import glob
import numpy as np
from tensorflow.keras.utils import to_categorical
from collections import defaultdict
from copy import deepcopy

# define problem properties
FASTA_RESIDUE_LIST = ["A", "D", "N", "R", "C", "E", "Q", "G", "H", "I",
                      "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
NB_RESIDUES = len(FASTA_RESIDUE_LIST)
RESIDUE_DICT = dict(zip(FASTA_RESIDUE_LIST, range(NB_RESIDUES)))
UPPER_LENGTH_LIMIT = 7168


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
