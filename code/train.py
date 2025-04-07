
import os
import numpy as np
from data_loading import read_fasta, fill_array_with_value, standardize_data
from architecture import unet_classifier
from custom_metrics import get_confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from sklearn.model_selection import train_test_split

UPPER_LENGTH_LIMIT = 1024
MASK_VALUE = 9999

def prepare_training_pair(prottrans_embed, label_vector, max_len=UPPER_LENGTH_LIMIT):
    X = np.expand_dims(fill_array_with_value(prottrans_embed, max_len, 0), axis=-1)
    Y = fill_array_with_value(label_vector, max_len, np.array([MASK_VALUE, MASK_VALUE]))
    return X, Y

def load_dataset(filepath):

    if os.path.exists('processed_dict_1024.pkl'):
        print("📂 Loading existing data_dict...")
        with open('processed_dict.pkl', "rb") as f:
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

    return X_train, X_val, Y_train, Y_val

def train_model(X_train, X_val, Y_train, Y_val, output_path="DisorderUnet.h5"):
    model = unet_classifier()
    callbacks = [
        ModelCheckpoint(output_path, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
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

if __name__ == "__main__":
    annotations = "../data/Alphafold RCL annotations.csv"
    model_output_path = "../data/models/RCLUnet.h5"

    print("🔄 Loading and processing data...")
    X_train, X_val, Y_train, Y_val = load_dataset(annotations)

    print("🧠 Training model...")
    model = train_model(X_train, X_val, Y_train, Y_val, output_path=model_output_path)

    Y_pred = model.predict(X_val, batch_size=1)
    cm = get_confusion_matrix(Y_val, Y_pred)




