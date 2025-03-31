import tensorflow as tf
tf.random.set_seed(1992)
from tensorflow.keras import backend as K

MASK_VALUE = 9999

def masked_acc(y_true, y_pred):
    mask = tf.reduce_any(tf.not_equal(y_true, MASK_VALUE), -1)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return K.mean(K.equal(K.argmax(y_true_masked, axis=-1), K.argmax(y_pred_masked, axis=-1)))


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

