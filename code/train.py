
import os
import json
import numpy as np
import pandas as pd 
from collections import defaultdict
from data_loading import prepare_train_set
from architecture import unet_classifier, cnn_classifier, lstm_classifier
from custom_metrics import get_confusion_matrix, masked_acc, masked_f1, get_histogram, masked_auc
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# takes split and scaled data, instantiates and trains a model based on encoding length and model architecture 
def train_model(X_train, X_val, Y_train, Y_val, encoding_length, model_type = 'unet', output_path="RCL_Unet.h5"):
    if model_type == 'unet':
        model = unet_classifier(encoding_length)
    elif model_type == 'cnn':
        model = cnn_classifier(encoding_length)
    elif model_type == 'lstm':
        model = lstm_classifier(encoding_length)
    else:
        print('invalid model type')
        return
    
    # monitors performance on validation set and stops training if performance doesnt improve after (patience = n) epochs 
    callbacks = [
        ModelCheckpoint(output_path, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    # trains model
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=32,
        epochs=20,
        shuffle=True,
        callbacks=callbacks
    )

    print(f"✅ Model training complete. Saved to: {output_path}")
    return model

# keeps track of version number 
def create_model_dir(base_dir="../models/runs"):
    os.makedirs(base_dir, exist_ok=True)

    existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run_")]
    next_run_id = len(existing_runs) + 1
    run_dir = os.path.join(base_dir, f"run_{next_run_id:03}")
    os.makedirs(run_dir)

    return run_dir

if __name__ == "__main__":

    # ======================= USER CONFIGURATION ========================

    # set parameters 
    model_type = 'lstm' # unet, lstm, cnn, 
    encoding = 'onehot' # onehot, blosum, prottrans, esm
    scaling = True

    # select training files 
    serpin_file = "../data/Alphafold RCL annotations.csv"
    non_serpin_file = "../data/non_serpin_train.csv"

    # ===================================================================

    if encoding == 'onehot':
        encoding_length = 21
    elif encoding == 'blosum':
        encoding_length = 20
    elif encoding == 'prottrans':
        encoding_length = 1024
    elif encoding == 'esm':
        encoding_length = 1280
    else:
        raise ValueError("invalid encoding")
         
    
    # call function to load, scale, and split data 
    print("loading and processing data...")
    run_dir = create_model_dir()
    X_train, X_val, Y_train, Y_val = prepare_train_set(serpin_file, non_serpin_file, encoding, scaling, run_dir)

    # call model training function 
    print("training model...")
    model_output_path = (f"{run_dir}/RCL_{model_type}.h5")
    model = train_model(X_train, X_val, Y_train, Y_val, encoding_length=encoding_length, model_type=model_type, output_path=model_output_path)

    # calculate performance on validation set, save paramters and metrics 
    Y_pred = model.predict(X_val, batch_size=32)

    metrics = {
        "model_type": model_type,
        "encoding": encoding,
        "encoding_length": encoding_length,
        "scaling": scaling,
        "val_accuracy": float(masked_acc(Y_val, Y_pred)),
        "val_f1": float(masked_f1(Y_val, Y_pred)),
        "val_macro_f1": float(masked_f1(Y_val, Y_pred, average='macro')),
        "val_auc": float(masked_auc(Y_val, Y_pred))
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # generating and saving figures 
    cm = get_confusion_matrix(Y_val, Y_pred, output_dir=f'{run_dir}')
    hist = get_histogram(Y_val, Y_pred, output_dir=f'{run_dir}')




