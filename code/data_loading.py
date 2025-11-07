import os
import sys 
import numpy as np
import pandas as pd
import json
import pickle
from collections import defaultdict

from tqdm import tqdm 

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from transformers import T5Model, T5EncoderModel, AutoTokenizer, AutoModel
import torch, re
# from bio_embeddings.embed import ProtTransT5XLU50Embedder

# define problem properties
# FASTA_RESIDUE_LIST = ["A", "D", "N", "R", "C", "E", "Q", "G", "H", "I",
#                       "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
# NB_RESIDUES = len(FASTA_RESIDUE_LIST)
# RESIDUE_DICT = dict(zip(FASTA_RESIDUE_LIST, range(NB_RESIDUES)))
UPPER_LENGTH_LIMIT = 1024
MASK_VALUE = 9999

# reads fasta file, returns python dictionary with encoded sequences 
def read_fasta_file(filepath: str, encoding):

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
        data_dict[protein_name]["sequence"] = resnames

    # apply encoding 
    encoded_dict = encode_dict(data_dict, encoding)
    
    print(len(data_dict), "proteins loaded")
    return encoded_dict

# reads csv file with labeled rcl information, returns python dictionary with encoded sequences and per-index labels 
def read_csv_file(filepath: str, encoding):

    data_dict = defaultdict(dict)

    # read csv
    df = pd.read_csv(filepath)
    
    # filter for length limit, get rid of rows without annotation 
    df = df[df["Sequence"].str.len() <= UPPER_LENGTH_LIMIT]
    if 'rcl_seq' in df.columns:
        df = df.dropna(subset=['rcl_seq']) 
    
    # iterate protein by protein 
    for _, row in df.iterrows():

        # reads id and sequence 
        protein_id = row["id"].strip()
        sequence = row["Sequence"].strip()
        data_dict[protein_id]["sequence"] = sequence

        # creating labels 
        rcl_label = np.full((UPPER_LENGTH_LIMIT, 2), [1, 0], dtype=np.float32)  # all non-RCL by default

        if 'rcl_seq' in df.columns: # if rcl_seq is not a column in the csv, it is the control dataset, and all labels remain non-RCL

            rcl_start = int(row['rcl_start'])
            rcl_end = int(row['rcl_end'])
            rcl_start_idx = max(0, rcl_start - 1)
            rcl_end_idx = min(len(sequence), rcl_end)  # do not exceed actual length

            for i in range(rcl_start_idx, rcl_end_idx):
                rcl_label[i] = [0, 1]

        # if sequence is shorter than max_length, apply mask to remaining indices
        for i in range(len(sequence), UPPER_LENGTH_LIMIT):
            rcl_label[i] = [9999, 9999]

        data_dict[protein_id]['label'] = rcl_label

    # apply encoding 
    encoded_dict = encode_dict(data_dict, encoding)
    
    return encoded_dict

# applies encoding to sequences in a dictionary 
def encode_dict(data_dict, encoding: str):

    encoding = encoding.lower() # ensure encoding name is lowercase

    # one-hot encoding
    if (encoding == 'onehot'):
        with open("../data/encodings/One_hot.json") as f:
            encoding_map = json.load(f)
        for protein_name in tqdm(data_dict, desc='Generating One-hot Encodings'):
            sequence = data_dict[protein_name]["sequence"]
            one_hot_encoded = [encoding_map.get(residue, encoding_map["X"]) for residue in sequence]
            data_dict[protein_name]["encoding"] = np.array(one_hot_encoded, dtype=np.float32)

    # BLOSUM62 based encoding
    elif (encoding == 'blosum'):
        with open("../data/encodings/BLOSUM62.json") as f:
            encoding_map = json.load(f)
        for protein_name in tqdm(data_dict, desc='Generating BLOSUM62 Encodings'):
            sequence = data_dict[protein_name]["sequence"]
            blosum_encoded = [encoding_map.get(residue, encoding_map["X"]) for residue in sequence]
            data_dict[protein_name]["encoding"] = np.array(blosum_encoded, dtype=np.float32)

    # ProtTransLM protein language model encoding 
    # elif (encoding == 'prottrans'):
    #     print("Loading ProtTrans model")   
    #     embedder = OfflineProtTransT5XLU50Embedder()
    #     for protein_name in tqdm(data_dict, desc='Calculating ProtTrans Features'):
    #         # uses unencoded sequence
    #         data_dict[protein_name]["encoding"] = embedder.embed(data_dict[protein_name]["sequence"])

    # ESM
    elif (encoding == 'esm2'):
        print("Loading ESM2 model")   
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to("cuda").eval() 
        for protein_name in tqdm(data_dict, desc='Calculating ProtTrans Features'):
            seq = data_dict[protein_name]["sequence"]
            tokens = tokenizer(" ".join(seq), return_tensors="pt")
            with torch.no_grad():
                out = model(**{k: v.to("cuda") for k, v in tokens.items()})
            # strip [CLS]/[EOS]
            data_dict[protein_name]["encoding"] = out.last_hidden_state[0, 1:-1]  

    # invalid encoding name
    else:
        print(f"{encoding} is an an invalid encoding")
        sys.exit()
    
    return data_dict

# pads out length of encoded protein sequence and label 
def prepare_training_pair(encoded_protein, label_vector, max_len=UPPER_LENGTH_LIMIT):
    X = np.expand_dims(fill_array_with_value(encoded_protein, max_len, 0), axis=-1)
    Y = fill_array_with_value(label_vector, max_len, np.array([MASK_VALUE, MASK_VALUE]))
    return X, Y

# trains new scaler and scales training data using scikit-learn's StandardScaler, saves scaler object 
def scale_training_data(X_train, X_val, save_dir):

    # num residues, length, dimensions, extra channel 
    N_train, T, D, C = X_train.shape
    N_val = len(X_val)
    print("scaling data")

    # StandardScaler only works on 2 dimensions, so remove last dimension and flatten to (N*T, D)
    X_train_flat = X_train.reshape(-1, D)
    X_val_flat = X_val.reshape(-1, D)

    # apply scaling 
    scaler = StandardScaler()
    X_train_flat_scaled = scaler.fit_transform(X_train_flat)
    X_val_flat_scaled = scaler.transform(X_val_flat)

    # after scaling, reshape to original shape 
    X_train = X_train_flat_scaled.reshape(N_train, T, D, C)
    X_val = X_val_flat_scaled.reshape(N_val, T, D, C)

    # save scaler file so it can be used when model is used 
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    return X_train, X_val

# applies existing scaler file to new data 
def apply_scaler(X, scaler_file):

    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f) 

    N, T, D, C = X.shape
    print("scaling data")

    # reshape to 2 dimensions, apply scaler, and restore original shape
    X_flat = X.reshape(-1, D)  # shape: (N*T, D)
    X_flat_scaled = scaler.transform(X_flat)
    X_scaled = X_flat_scaled.reshape(N, T, D, C)

    return X_scaled

# reads labeled csv, splits data, returns scaled data
def prepare_train_set(serpin_file, non_serpin_file , encoding='onehot', scaling=True, run_dir=''):

    # if filepath.endswith('.pkl'):
    #     print("📂 Loading existing data_dict...")
    #     with open(filepath, "rb") as f:
    #         data_dict = pickle.load(f)
    # else:
    
    print("reading from csv")
    serpin_dict = read_csv_file(serpin_file, encoding)
    non_serpin_dict = read_csv_file(non_serpin_file, encoding)
    
    data_dict = defaultdict(dict)
    data_dict.update(serpin_dict)
    data_dict.update(non_serpin_dict)

    X = []
    Y = []

    for protein_id in data_dict:

        encoded_protein = data_dict[protein_id]["encoding"]
        label_vector = data_dict[protein_id]["label"]

        x, y = prepare_training_pair(encoded_protein, label_vector)
        X.append(x)
        Y.append(y)

    X = np.stack(X)
    Y = np.stack(Y)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    if (scaling):
        X_train, X_val = scale_training_data(X_train, X_val, run_dir)

    return X_train, X_val, Y_train, Y_val

# reads labeled csv, doesn't split data, returns scaled data + list of IDs and sequences  
def prepare_test_set(filepath: str, encoding: str, scaler_file):

    print("reading from csv")
    data_dict= read_csv_file(filepath, encoding)

    X = []
    Y = []
    id_list = []
    sequences = []

    for protein_id in data_dict:

        encoded_protein = data_dict[protein_id]["encoding"]
        label_vector = data_dict[protein_id]["label"]
        sequence = data_dict[protein_id]["sequence"]

        x, y = prepare_training_pair(encoded_protein, label_vector)
        X.append(x)
        Y.append(y)
        id_list.append(protein_id)
        sequences.append(sequence)

    X = np.stack(X)
    Y = np.stack(Y)

    X_scaled = apply_scaler(X, scaler_file)

    return X_scaled, Y, id_list, sequences

# reads unlabeled fasta file, returns scaled data and list of IDs 
def prepare_unlabeled_set(filepath: str, encoding: str, scaler_file):

    print("reading from fasta")
    data_dict= read_fasta_file(filepath, encoding)

    X = []
    id_list = []

    for protein_id in data_dict:

        encoded_protein = data_dict[protein_id]["encoding"]

        x = np.expand_dims(fill_array_with_value(encoded_protein, UPPER_LENGTH_LIMIT, 0), axis=-1)
        X.append(x)
        id_list .append(protein_id)

    X = np.stack(X)

    X_scaled = apply_scaler(X, scaler_file)

    return X_scaled, id_list

# fills array with value, considering shape 
def fill_array_with_value(array: np.array, length_limit: int, value):

    filler = value * np.ones((length_limit - array.shape[0], array.shape[1]), array.dtype)
    filled_array = np.concatenate((array, filler))

    return filled_array


# local instance of ProtTransLM 
# class OfflineProtTransT5XLU50Embedder(ProtTransT5XLU50Embedder):
#     # Use an offline model directory
#     def __init__(self, **kwargs):
#         self.necessary_directories = []
#         super().__init__(model_directory=os.path.join('../models', "prot_t5_xl_uniref50"))
#         self._half_precision_model = False

#     def get_model(self):
#         if not self._decoder:
#             print('Using T5EncoderModel')
#             # model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
#             model = T5EncoderModel.from_pretrained(self._model_directory)
#         else:
#             print('Using T5Model')
#             model = T5Model.from_pretrained(self._model_directory)
#         return model
    
# huggingface prottranslm
# class ProtEmbedder:
#     def __init__(self, model_name="Rostlab/prot_t5_xl_uniref50", device=None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
#         self.model = AutoModel.from_pretrained(model_name).to(self.device)
#         self.model.eval()

#     def preprocess(self, sequence: str):
#         # Replace ambiguous residues and insert spaces
#         sequence = re.sub(r"[UZOB]", "X", sequence)
#         return " ".join(list(sequence))

#     def embed(self, sequence: str, per_residue=True):
#         seq = self.preprocess(sequence)
#         ids = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
#         input_ids = ids["input_ids"].to(self.device)
#         attention_mask = ids["attention_mask"].to(self.device)

#         with torch.no_grad():
#             outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#             embeddings = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)

#         # Remove special tokens ([CLS], [SEP]) and padding
#         valid_len = attention_mask[0].sum().item()
#         embeddings = embeddings[0, 1:valid_len-1]

#         if per_residue:
#             return embeddings.cpu()  # shape: (L, hidden_dim)
#         else:
#             return embeddings.mean(dim=0).cpu()  # shape: (hidden_dim,)
    
# def standardize_data(data_dict: dict):

#     mean = np.load(os.path.join("data_stats", "train_mean_prottrans.npy"))
#     std = np.load(os.path.join("data_stats", "train_std_prottrans.npy"))

#     for key in data_dict.keys():
#         data_dict[key]["prottrans"] = (data_dict[key]["prottrans"] - mean) / std

#     return data_dict

# def fill_with_zeros(data: dict, max_sequence_length: int):
#     data_copy = deepcopy(data)
#     for key, values in data_copy.items():
#         if len(values) == UPPER_LENGTH_LIMIT:
#             continue
#         elif len(values) > UPPER_LENGTH_LIMIT:
#             data_copy[key] = values[:UPPER_LENGTH_LIMIT]
#             continue
#         data_copy[key] = fill_array_with_value(values, max_sequence_length, 0)

#     return data_copy