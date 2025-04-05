
import os
import numpy as np
from data_loading import read_fasta, read_disorder_labels, fill_array_with_value, standardize_data
from architecture import unet_classifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

UPPER_LENGTH_LIMIT = 1000
MASK_VALUE = 9999

def prepare_training_pair(prottrans_embed, label_vector, max_len=UPPER_LENGTH_LIMIT):
    X = np.expand_dims(fill_array_with_value(prottrans_embed, max_len, 0), axis=-1)
    Y = fill_array_with_value(label_vector, max_len, np.array([MASK_VALUE, MASK_VALUE]))
    return X, Y

def load_dataset(fasta_path, labels_path):
    data_dict = read_fasta(fasta_path)
    labels_dict = read_disorder_labels(labels_path)

    X, Y = [], []
    for protein_name in data_dict:
        if protein_name not in labels_dict:
            continue

        # Assume ProtTrans features already computed and available
        prottrans_embed = data_dict[protein_name]["prottrans"]
        label_vector = labels_dict[protein_name]

        x, y = prepare_training_pair(prottrans_embed, label_vector)
        X.append(x)
        Y.append(y)

    X = np.stack(X)
    Y = np.stack(Y)
    return X, Y

def train_model(X, Y, output_path="DisorderUnet.h5"):
    model = unet_classifier()
    callbacks = [
        ModelCheckpoint(output_path, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]
    model.fit(
        X, Y,
        validation_split=0.1,
        batch_size=1,
        epochs=20,
        shuffle=True,
        callbacks=callbacks
    )
    print(f"✅ Model training complete. Saved to: {output_path}")

if __name__ == "__main__":
    fasta_file = "../data/inputs/train.fasta"
    label_file = "../data/inputs/train_labels.caid"
    model_output_path = "../data/models/DisorderUnet.h5"

    print("🔄 Loading and processing data...")
    X, Y = load_dataset(fasta_file, label_file)

    print("🧠 Training model...")
    train_model(X, Y, output_path=model_output_path)



