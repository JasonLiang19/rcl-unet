import numpy as np
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
from custom_metrics import masked_acc, mcc_cc_loss, mcc_metric, get_histogram, get_confusion_matrix, masked_f1, masked_auc
from architecture import unet_classifier
import pickle
from train import prepare_training_pair
from sklearn.preprocessing import StandardScaler
import json

print(tf.config.list_physical_devices('GPU'))

run_dir = "../data/models/runs/run_008/"
results_dir = os.path.join(run_dir, 'results')
os.makedirs(results_dir)

model = unet_classifier()
model.load_weights(os.path.join(run_dir, 'RCL_Unet.h5'))

with open("../data/test_data.pkl", "rb") as f:
    data_dict = pickle.load(f)

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
        "test_macro_f1": float(masked_f1(Y_test, Y_pred, average='macro')),
        "test_auc": float(masked_auc(Y_test, Y_pred))
    }
with open(os.path.join(results_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)


