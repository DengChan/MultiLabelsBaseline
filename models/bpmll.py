import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import GRU, CuDNNGRU
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K


def loss_fun(y_true, y_pred):
    shape = K.shape(y_true)
    y_i = K.equal(y_true, tf.ones(shape, dtype=tf.float32))
    y_not_i = K.equal(y_true, tf.zeros(shape, dtype=tf.float32))

    # get indices to check
    truth_matrix = K.to_float(pairwise_and(y_i, y_not_i))

    # calculate all exp'd differences
    # through and with truth_matrix, we can get all c_i - c_k(appear in the paper)
    sub_matrix = pairwise_sub(y_pred, y_pred)
    exp_matrix = K.exp(tf.negative(sub_matrix))

    # check which differences to consider and sum them
    sparse_matrix = K.multiply(exp_matrix, truth_matrix)
    sums = tf.reduce_sum(sparse_matrix, axis=[1, 2])

    # get normalizing terms and apply them
    y_i_sizes = K.reduce_sum(tf.to_float(y_i), axis=1)
    y_i_bar_sizes = K.reduce_sum(tf.to_float(y_not_i), axis=1)
    normalizers = K.multiply(y_i_sizes, y_i_bar_sizes)
    loss = tf.divide(sums, normalizers)

    return loss


def pairwise_sub(a, b):
    column = K.expand_dims(a, 2)
    row = K.expand_dims(b, 1)
    return K.subtract(column, row)


def pairwise_and(a, b):
    column = K.expand_dims(a, 2)
    row = K.expand_dims(b, 1)
    return K.logical_and(column, row)


def BPMLLModel(optimizer, num_classes):
    model = Sequential()
    #model.add(keras.layers.Embedding(max_words+1, embedding_size, input_length=sequence_length, trainable=True))
    model.add(keras.layers.Dense(800, activation='tanh', kernel_regularizer=regularizers.l2(0.01)))
    model.add(keras.layers.Dense(num_classes, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['categorical_accuracy'])
    return model

