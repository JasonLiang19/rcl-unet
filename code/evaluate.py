import numpy as np
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
from custom_metrics import masked_acc, mcc_cc_loss, mcc_metric, get_histogram, get_confusion_matrix, masked_f1, masked_auc
from architecture import unet_classifier
import pickle
from train import prepare_training_pair, UPPER_LENGTH_LIMIT
from sklearn.preprocessing import StandardScaler
import json
from data_loading import read_seq_file, fill_array_with_value
print(tf.config.list_physical_devices('GPU'))

def evaluate(data_dict, results_dir):
    # serpins 
    X = []
    Y = []

    for protein_name in data_dict:

        # Assume ProtTrans features already computed and available
        encoded_protein = data_dict[protein_name]["encoding"]
        label_vector = data_dict[protein_name]["label"]

        x, y = prepare_training_pair(encoded_protein, label_vector)
        X.append(x)
        Y.append(y)

    X_test = np.stack(X)
    Y_test = np.stack(Y)

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
    Y_pred = model.predict(X_test)

    # metrics 
    print(masked_acc(Y_test, Y_pred))
    get_histogram(Y_test, Y_pred, results_dir)
    get_confusion_matrix(Y_test, Y_pred, results_dir)

    metrics = {
            "test_accuracy": float(masked_acc(Y_test, Y_pred)),
            "test_f1": float(masked_f1(Y_test, Y_pred)),
            "test_macro_f1": float(masked_f1(Y_test, Y_pred, average='macro'))
        }
    valid_mask = ~(np.all(Y_test == 9999, axis=-1))
    # Convert softmax/logits to class labels (0 or 1)
    Y_test_classes = np.argmax(Y_test, axis=-1)  # shape: (N, T)
    num_rcl = np.sum((Y_test_classes == 1) & valid_mask) # Check if there are any predicted RCL positions (i.e., predicted class = 1)
    print('geegoo')
    print(np.sum((Y_test_classes == 1) & valid_mask))
    print(np.sum((Y_test_classes == 0) & valid_mask))
    if num_rcl > 0:
        metrics["test_auc"] = float(masked_auc(Y_test, Y_pred))
    
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

def evaluate_sequence(sequence, run_dir = "../data/models/runs/run_007/"):
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
    model = unet_classifier(parameters["encoding_length"])
    model.load_weights(os.path.join(run_dir, 'RCL_Unet.h5')) # loads specific model from current run 
    Y_pred = model.predict(X_test)
    return Y_pred[0, :, 1]


def main():
    run_dir = "../data/models/runs/run_007/"
    serpin_dir = os.path.join(run_dir, 'results_serpin')
    non_serpin_dir = os.path.join(run_dir, 'results_non_serpin')
    os.makedirs(serpin_dir, exist_ok=True)
    os.makedirs(non_serpin_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'metrics.json')) as f:
        parameters = json.load(f) 

    model = unet_classifier(parameters["encoding_length"])
    model.load_weights(os.path.join(run_dir, 'RCL_Unet.h5')) # loads specific model from current run 

    if parameters["encoding"] == 'prottrans':
        serpin_dict = read_seq_file("../data/test_data.pkl", 'prottrans')
        non_serpin_dict = read_seq_file("../data/non_serpin_test.csv", 'prottrans')
    elif parameters["encoding"] == 'onehot':
        serpin_dict = read_seq_file("../data/Uniprot Test Set.csv", 'onehot')
        non_serpin_dict = read_seq_file("../data/non_serpin_test.csv", 'onehot')
    elif parameters["encoding"] == 'blosum':
        data_dict = read_seq_file("../data/Uniprot Test Set.csv", 'blosum')

    evaluate(serpin_dict, serpin_dir)
    evaluate(non_serpin_dict, non_serpin_dir)

if __name__ == "__main__":
    main()


