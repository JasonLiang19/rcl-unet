import tensorflow as tf
tf.random.set_seed(1992)
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

MASK_VALUE = 9999

def masked_acc(y_true, y_pred):
    mask = tf.reduce_any(tf.not_equal(y_true, MASK_VALUE), -1)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return K.mean(K.equal(K.argmax(y_true_masked, axis=-1), K.argmax(y_pred_masked, axis=-1)))

def masked_f1(y_true, y_pred, average='binary'):

     # Flatten to 2D: (N * 1000, 2)
    y_true = np.reshape(y_true, (-1, 2))
    y_pred = np.reshape(y_pred, (-1, 2))

    # Mask out padded values
    valid_mask = ~np.all(y_true == MASK_VALUE, axis=-1)
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]

    # Convert to class indices
    y_true_idx = np.argmax(y_true_clean, axis=-1)
    y_pred_idx = np.argmax(y_pred_clean, axis=-1)

    # Compute F1
    return f1_score(y_true_idx, y_pred_idx, average=average)

def mcc_cc_loss(y_true_org, y_pred_org):
    mask = tf.reduce_any(tf.not_equal(y_true_org, MASK_VALUE), -1)
    y_true = tf.boolean_mask(y_true_org, mask)
    y_pred = tf.boolean_mask(y_pred_org, mask)

    tp = K.sum(K.cast(y_true * y_pred, 'float32'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float32'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float32'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float32'), axis=0)

    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + K.epsilon()
    mcc = (tp * tn - fp * fn) / K.sqrt(denom)
    return K.mean(K.categorical_crossentropy(y_true, y_pred)) - K.mean(mcc)


def mcc_metric(y_true_org, y_pred_prg):
    mask = tf.reduce_any(tf.not_equal(y_true_org, MASK_VALUE), -1)
    y_true = tf.boolean_mask(y_true_org, mask)
    y_pred = tf.boolean_mask(y_pred_prg, mask)

    nb_classes = tf.cast(K.shape(y_pred)[1], tf.float32)
    threshold = tf.cast(tf.divide(tf.constant(1.0), nb_classes), tf.float32)
    predicted = tf.cast(tf.greater(y_pred, threshold), tf.float32)
    true_pos = tf.math.count_nonzero(predicted * y_true)
    true_neg = tf.math.count_nonzero((predicted - 1) * (y_true - 1))
    false_pos = tf.math.count_nonzero(predicted * (y_true - 1))
    false_neg = tf.math.count_nonzero((predicted - 1) * y_true)
    x = tf.cast((true_pos + false_pos) * (true_pos + false_neg)
      * (true_neg + false_pos) * (true_neg + false_neg), tf.float32)
    return tf.cast((true_pos * true_neg) - (false_pos * false_neg), tf.float32) / tf.sqrt(x)

def get_confusion_matrix(y_true, y_pred, output_dir='cm.png'):
    
    # Mask out padding
    valid_mask = ~np.all(y_true == MASK_VALUE, axis=-1)
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]

    # Convert to class indices
    y_true_idx = np.argmax(y_true_clean, axis=-1)
    y_pred_idx = np.argmax(y_pred_clean, axis=-1)

    # compute confusion matrix
    cm = confusion_matrix(y_true_idx, y_pred_idx)

    # plot 
    labels = [
        ["True non-RCL", "False RCL"],
        ["False non-RCL", "True RCL"]
    ]

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True, cbar=False,
                xticklabels=["Pred non-RCL", "Pred RCL"],
                yticklabels=["True non-RCL", "True RCL"])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    
    plt.savefig(output_dir)

def get_histogram(y_true, y_pred, output_dir='hist.png'):

    num_proteins = y_true.shape[0]
    errors_per_protein = []

    for i in range(num_proteins):
        yt = y_true[i]
        yp = y_pred[i]

        # Create mask for valid positions
        valid_mask = ~np.all(yt == MASK_VALUE, axis=-1)

        # Apply mask
        yt_clean = yt[valid_mask]
        yp_clean = yp[valid_mask]

        # Convert to class indices
        yt_idx = np.argmax(yt_clean, axis=-1)
        yp_idx = np.argmax(yp_clean, axis=-1)

        # Count mismatches
        num_wrong = np.sum(yt_idx != yp_idx)
        errors_per_protein.append(num_wrong)

    max_errors = max(errors_per_protein)
    bins = np.arange(0, max_errors + 2)  # +2 so the last bin includes max_errors

    plt.figure(figsize=(8, 5))
    plt.hist(errors_per_protein, bins=bins, color='skyblue', edgecolor='black', align='left')
    plt.xticks(bins)  # Show every integer tick
    plt.xlabel("Number of incorrect residues per protein")
    plt.ylabel("Number of proteins")
    plt.title("Residue Misclassifications per Protein")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()