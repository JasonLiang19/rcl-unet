import argparse
import pandas as pd
import os
import time
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import numpy as np
import torch
import tensorflow as tf
from transformers import T5Model, T5EncoderModel
from bio_embeddings.embed import ProtTransT5XLU50Embedder
from data_loading import UPPER_LENGTH_LIMIT, FASTA_RESIDUE_LIST, read_fasta, standardize_data, fill_array_with_value, fill_with_zeros
from architecture import build_ensemble

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DisorderUnetLM prediction."
    )
    parser.add_argument(
        "--input_fasta",
        "-i",
        type=str,
        default="../data/inputs/CAID2-test.fasta",
        help="Path to the input FASTA file",
    )
    parser.add_argument(
        "--models_folder",
        "-m",
        type=str,
        default="../data/models",
        help="Path to the folder with models",
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        default="../results/disorder",
        help="Path to the folder to generate results into",
    )

    return parser.parse_args()

args = parse_args()

os.makedirs(args.output_folder, exist_ok=True)

class OfflineProtTransT5XLU50Embedder(ProtTransT5XLU50Embedder):
    # Use an offline model directory
    def __init__(self, **kwargs):
        self.necessary_directories = []
        super().__init__(model_directory=os.path.join(args.models_folder, "prot_t5_xl_uniref50"), **kwargs)
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


print("Loading ProtTrans model")   
embedder = OfflineProtTransT5XLU50Embedder()


def save_prediction_to_file(residues_one_hot: np.array, pred_c: np.array, protein_name: str):
    sequence_length = len(residues_one_hot)

    resnames = [FASTA_RESIDUE_LIST[idx] for idx in np.argmax(residues_one_hot, axis=-1)]

    def get_prob(pred):
        return [val[1] for val in pred]

    def get_disorder(pred):
        return [idx for idx in np.argmax(pred, axis=-1)]

    probs = get_prob(pred_c[:sequence_length])
    preds = get_disorder(pred_c[:sequence_length])

    with open(os.path.join(args.output_folder, protein_name + ".caid"), "w") as file_writer:
        file_writer.write(f">{protein_name}\n")
        for idx, (resname, prob, pred) in enumerate(zip(resnames, probs, preds), 1):
            file_writer.write(f"{idx}\t{resname}\t{prob}\t{pred}\n")


def calculate_prottrans_features(data_dict: dict) -> dict:
    for protein_name in data_dict:
        sequence = "".join([FASTA_RESIDUE_LIST[idx] for idx in np.argmax(data_dict[protein_name]["fasta"], axis=-1)])
        data_dict[protein_name]["prottrans"] = embedder.embed(sequence)

    return data_dict


def main():
    print("Loading input data")
    data_dict = read_fasta(args.input_fasta)

    print("Calculating ProtTrans features")
    data_dict = calculate_prottrans_features(data_dict)

    print("Data standardization")
    data_dict = standardize_data(data_dict)

    print("Loading DisorderUnetLM model")
    model = build_ensemble()
    model.load_weights(os.path.join(args.models_folder, "DisorderUnet.h5"))
    model = tf.function(model) # speed up with graph execution

    with open(os.path.join(args.output_folder, "..", "timings.csv"), "w") as timing_writer:
        timing_writer.write(f"# Running DisorderUnetLM, started {datetime.now()}\n")
        timing_writer.write("sequence,milliseconds\n")

        for protein_name, data in data_dict.items():
            print("Processing protein", protein_name)

            protein_length = len(list(data.values())[0])
            nb_windows = np.ceil(protein_length / UPPER_LENGTH_LIMIT)
            if nb_windows > 1:
                print(f"Sequence longer than {UPPER_LENGTH_LIMIT} residues! Predicting only the first {UPPER_LENGTH_LIMIT} residues")

            start_time = time.time()
            filled_data = fill_with_zeros(data, UPPER_LENGTH_LIMIT)
            input_data = np.expand_dims(filled_data["prottrans"], axis=0)
            prediction = np.array(model(input_data))
            elapsed_time = int(1000 * (time.time() - start_time))
            timing_writer.write(f"{protein_name},{elapsed_time}\n")
            print(f"Elapsed milliseconds: {elapsed_time}")

            if nb_windows > 1:
                prediction = fill_array_with_value(prediction, protein_length, 0)

            save_prediction_to_file(data["fasta"], prediction, protein_name)
        print("Finished")


if __name__ == '__main__':
    main()
