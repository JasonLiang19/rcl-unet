
import os
import json
import numpy as np
from data_loading import read_fasta, fill_array_with_value, standardize_data
from architecture import unet_classifier
from custom_metrics import get_confusion_matrix, masked_acc, masked_f1
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

UPPER_LENGTH_LIMIT = 1024
MASK_VALUE = 9999

def prepare_training_pair(prottrans_embed, label_vector, max_len=UPPER_LENGTH_LIMIT):
    X = np.expand_dims(fill_array_with_value(prottrans_embed, max_len, 0), axis=-1)
    Y = fill_array_with_value(label_vector, max_len, np.array([MASK_VALUE, MASK_VALUE]))
    return X, Y

def load_dataset(filepath):

    if os.path.exists(filepath):
        print("📂 Loading existing data_dict...")
        with open(filepath, "rb") as f:
            data_dict = pickle.load(f)
    else:
        print("reading from csv")
        data_dict = read_fasta(filepath)

    X = []
    Y = []

    for protein_name in data_dict:

        # Assume ProtTrans features already computed and available
        prottrans_embed = data_dict[protein_name]["prottrans"]
        label_vector = data_dict[protein_name]["label"]

        x, y = prepare_training_pair(prottrans_embed, label_vector)
        X.append(x)
        Y.append(y)

    X = np.stack(X)
    Y = np.stack(Y)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)

    return X_train, X_val, Y_train, Y_val

def train_model(X_train, X_val, Y_train, Y_val, output_path="RCL_Unet.h5"):
    model = unet_classifier()
    callbacks = [
        ModelCheckpoint(output_path, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=1,
        epochs=20,
        shuffle=True,
        callbacks=callbacks
    )
    print(f"✅ Model training complete. Saved to: {output_path}")
    return model

# keeps track of version 
def create_run_dir(base_dir="../data/models/runs"):
    os.makedirs(base_dir, exist_ok=True)

    existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run_")]
    next_run_id = len(existing_runs) + 1
    run_dir = os.path.join(base_dir, f"run_{next_run_id:03}")
    os.makedirs(run_dir)

    return run_dir

if __name__ == "__main__":

    run_dir = create_run_dir()

    data = "../data/processed_dict_1024.pkl"
    model_output_path = (f"{run_dir}/RCL_Unet.h5")

    print("🔄 Loading and processing data...")
    X_train, X_val, Y_train, Y_val = load_dataset(data)

    # np.savez_compressed("../data/val_data.npz", X_val=X_val, Y_val=Y_val)

    print("🧠 Training model...")
    model = train_model(X_train, X_val, Y_train, Y_val, output_path=model_output_path)

    Y_pred = model.predict(X_val, batch_size=1)

    metrics = {
        "val_accuracy": float(masked_acc(Y_val, Y_pred)),
        "val_f1": float(masked_f1(Y_val, Y_pred)),
        "val_macro_f1": float(masked_f1(Y_val, Y_pred, average='macro'))
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    cm = get_confusion_matrix(Y_val, Y_pred, output_dir=f'{run_dir}/confusion matrix.png')




