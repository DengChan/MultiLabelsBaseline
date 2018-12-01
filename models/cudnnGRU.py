import tensorflow.keras as keras
from tensorflow.keras.layers import GRU, CuDNNGRU
from tensorflow.keras.models import Sequential


def GRUModel(max_words, embedding_size, sequence_length ,optimizer,num_classes):
    model = Sequential()
    model.add(keras.layers.Embedding(max_words+1, embedding_size, input_length=sequence_length))
    model.add(CuDNNGRU(num_classes))
    model.add(keras.layers.Activation('sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['categorical_accuracy'])
    return model
