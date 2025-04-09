import numpy as np
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
from custom_metrics import masked_acc, mcc_cc_loss, mcc_metric, get_histogram
from architecture import unet_classifier

model = unet_classifier()
model.load_weights("../data/models/runs/run_003/RCL_Unet.h5")

data = np.load("../data/val_data.npz")
X_val = data["X_val"]
Y_val = data["Y_val"]

Y_pred = model.predict(X_val, batch_size=1)

print(masked_acc(Y_val, Y_pred))
get_histogram(Y_val, Y_pred)