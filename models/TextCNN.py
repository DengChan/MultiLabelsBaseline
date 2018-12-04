import tensorflow.keras as keras
from tensorflow.keras.models import Sequential


def CNNModel(filter_sizes, num_filters, num_words, embedding_size, sequence_length, optimizer, num_classes):
    convs = []
    # shape 不包括Batchsize
    x_input = keras.layers.Input(shape=[sequence_length], name="x_input")
    # 参数：字典长度，embedding维度，
    # 变为1000x100
    embedding_layer = keras.layers.Embedding(num_words + 1, embedding_size,
                                          input_length=sequence_length,
                                          trainable=True)
    # 卷积
    for fz in filter_sizes:
        emb = embedding_layer(x_input)
        print("embedding shape: ", emb.get_shape())
        conv = keras.layers.Conv1D(num_filters, fz, padding="valid", activation='tanh')(emb)
        pool = keras.layers.MaxPool1D(sequence_length - fz + 1, padding='valid')(conv)
        pool = keras.layers.Flatten()(pool)
        # 平铺到一维
        convs.append(pool)

    # 横向连接，得到的还是一维
    merged = keras.layers.concatenate(convs, axis=1)

    # dropout
    merged = keras.layers.Dropout(0.5)(merged)
    # 全连接层
    dense = keras.layers.Dense(512, activation='tanh')(merged)
    output = keras.layers.Dense(num_classes, activation='sigmoid')(dense)
    model = keras.models.Model(x_input, output)
    # 优化器
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['categorical_accuracy'])

    return model
