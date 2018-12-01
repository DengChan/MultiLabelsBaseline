import os
import time
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from utils.LoadData import preprocess
from utils.config import conf
from models.cudnnGRU import GRUModel
from utils.Estimate import Metrics

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    with sess.as_default():
        x_train, y_train, x_dev, y_dev, word_index = preprocess(conf.train_path, conf.test_path)
        # optimizer
        optimizer = keras.optimizers.Adam(lr=1e-3, decay=1e-6)
        num_words = min(conf.max_words, len(word_index))
        if conf.model_type == 'GRU':
            model = GRUModel(num_words, conf.embedding_size, conf.sequence_length, optimizer, conf.num_classes)

        metric = Metrics()
        print("开始训练 ....")
        model.fit(x_train, y_train, batch_size=64, epochs=100,
                  callbacks=[metric, keras.callbacks.TensorBoard(log_dir='log/')],
                  validation_data=[x_dev, y_dev], shuffle=True)
