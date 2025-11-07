import numpy as np
import pandas as pd
import os
import glob
from tensorflow.keras.models import load_model
import tensorflow as tf
from custom_metrics import masked_acc, mcc_cc_loss, mcc_metric, get_histogram, get_confusion_matrix, masked_f1, masked_auc
from architecture import unet_classifier, cnn_classifier, lstm_classifier
import pickle
from sklearn.preprocessing import StandardScaler
import json
from data_loading import UPPER_LENGTH_LIMIT, fill_array_with_value, prepare_test_set
from utils import csv_to_fasta


print(tf.config.list_physical_devices('GPU'))

# assumes data is already scaled, runs the model and returns predictions  
def run_model(X, model_type, encoding_length, model_file):
    if model_type == 'unet':
        model = unet_classifier(encoding_length)
    elif model_type == 'cnn':
        model = cnn_classifier(encoding_length)
    elif model_type == 'lstm':
        model = lstm_classifier(encoding_length)

    model.load_weights(model_file) # loads specific model from current run 

    # TODO postprocessing 

    return model.predict(X)


# def predict_rcl(df, results_dir, model_dir):
#     return 

# evaluates the performance of a model on a dataset, saving metrics and figures 
def evaluate(filepath, encoding, results_dir, run_dir):

    # load data and parameters
    X_test, Y_test, ids, sequences = prepare_test_set(filepath, encoding, os.path.join(run_dir, 'scaler.pkl'))
    with open(os.path.join(run_dir, 'metrics.json')) as f:
        parameters = json.load(f) 

    # load model and perform inference 
    model_files = glob.glob(os.path.join(run_dir, '*.h5')) # agnostic to model file name, only looks for .h5 format 
    if len(model_files) == 0:
        raise FileNotFoundError(f"No .h5 model file found in: {run_dir}")
    Y_pred = run_model(X_test, parameters["model_type"], parameters["encoding_length"], model_files[0])
    
    # calculate and save metrics 
    print(masked_acc(Y_test, Y_pred))
    
    metrics = {
            "test_accuracy": float(masked_acc(Y_test, Y_pred)),
            "test_f1": float(masked_f1(Y_test, Y_pred)),
            "test_macro_f1": float(masked_f1(Y_test, Y_pred, average='macro'))
        }

    # determine if auc score can be calculated, as the number of (true positive + false negative) can't be 0  
    valid_mask = ~(np.all(Y_test == 9999, axis=-1))
    Y_test_classes = np.argmax(Y_test, axis=-1)  # shape: (N, T)
    num_rcl = np.sum((Y_test_classes == 1) & valid_mask) # Check if there are any predicted RCL positions (i.e., predicted class = 1)
    if num_rcl > 0:
        metrics["test_auc"] = float(masked_auc(Y_test, Y_pred))
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # save id, sequence, predicted_rcl_indices, and num_mistakes to csv
    records = []
    Y_pred_classes = np.argmax(Y_pred, axis=-1)  #     # Convert predictions to class labels
    for i, protein_id in enumerate(ids):
        sequence = sequences[i]
        pred_class = Y_pred_classes[i]  # shape: (T,)
        true_class = Y_test_classes[i]
        mask = valid_mask[i]

        # Get predicted RCL indices (positions where class == 1)
        pred_rcl_indices = np.where((pred_class == 1) & mask)[0].tolist()
        true_rcl_indices = np.where((true_class == 1) & mask)[0].tolist()

        records.append({
            "protein_id": protein_id,
            "sequence": sequence,
            "predicted_rcl_indices": pred_rcl_indices,
            "num_mistakes": np.sum(pred_class[mask] != true_class[mask])
        })
    df_results = pd.DataFrame(records)

    # save predictions to csv, generate histogram and confusion matrix 
    csv_path = os.path.join(results_dir, "rcl_predictions.csv")
    df_results.to_csv(csv_path, index=False)
    get_histogram(Y_test, Y_pred, results_dir)
    get_confusion_matrix(Y_test, Y_pred, results_dir)

    # fasta_path = os.path.join(results_dir, "rcl_predictions.fasta")
    # csv_to_fasta(csv_path, fasta_path, True)

# prediction of single sequence, for we app 
def evaluate_sequence(sequence, run_dir = "../models/runs/run_007/"):
    # serpins 
    X = []
    # Y = []

    with open("../data/encodings/One_hot.json") as f:
            encoding_map = json.load(f)
            one_hot_encoded = [encoding_map.get(residue, encoding_map["X"]) for residue in sequence]
            x = np.array(one_hot_encoded, dtype=np.float32)
            print(f"x:{x.shape}")

    # for protein_name in data_dict:

    #     # Assume ProtTrans features already computed and available
    #     encoded_protein = data_dict[protein_name]["encoding"]
    #     label_vector = data_dict[protein_name]["label"]

    #     x, y = prepare_training_pair(encoded_protein, label_vector)
    #     X.append(x)
    #     Y.append(y)
    x = np.expand_dims(fill_array_with_value(x, UPPER_LENGTH_LIMIT, 0), axis=-1)
    print(f"x exapnded:{x.shape}")
    X.append(x)

    X_test = np.stack(X)
    # Y_test = np.stack(Y)
    print(f"X_test stacked:{X_test.shape}")

    # scaling data
    if os.path.isfile(os.path.join(run_dir, 'scaler.pkl')): # if there is a scaler file in directory 
        with open(os.path.join(run_dir, 'scaler.pkl'), "rb") as f:
            scaler = pickle.load(f) 

        N, T, D, C = X_test.shape
        print("scaling data")

        # Remove last dimension and flatten to (N*T, D)
        X_test_flat = X_test.reshape(-1, D)  # shape: (N*1024, 1024)
        X_test_flat_scaled = scaler.transform(X_test_flat)
        X_test = X_test_flat_scaled.reshape(N, T, D, C)

    # perform inference
    with open(os.path.join(run_dir, 'metrics.json')) as f:
        parameters = json.load(f) 

    if parameters["model_type"] == 'unet':
        model = unet_classifier(parameters["encoding_length"])
    elif parameters["model_type"] == 'cnn':
        model = cnn_classifier(parameters["encoding_length"])
    elif parameters["model_type"] == 'lstm':
        model = lstm_classifier(parameters["encoding_length"])

    model.load_weights(os.path.join(run_dir, 'RCL_Unet.h5')) # loads specific model from current run 
    Y_pred = model.predict(X_test)
    return Y_pred[0, :, 1]


def main():
    
    # ======================= USER CONFIGURATION ========================

    # select model directory 
    run_dir = "../models/runs/run_013/"

    # select test dataset files
    serpin_test_set = "../data/Uniprot Test Set.csv"
    non_serpin_test_set = "../data/non_serpin_test.csv"

    # ===================================================================
    
    serpin_test_dir = os.path.join(run_dir, 'results_serpin')
    non_serpin_test_dir = os.path.join(run_dir, 'results_non_serpin')
    os.makedirs(serpin_test_dir, exist_ok=True)
    os.makedirs(non_serpin_test_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'metrics.json')) as f:
        parameters = json.load(f) 

    evaluate(serpin_test_set, parameters["encoding"], serpin_test_dir, run_dir)
    evaluate(non_serpin_test_set, parameters["encoding"],  non_serpin_test_dir, run_dir)

if __name__ == "__main__":
    main()


