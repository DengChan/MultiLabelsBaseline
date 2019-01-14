import os
import time
import json
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from utils.LoadData import preprocess
from utils.config import conf
from models.cudnnGRU import GRUModel
from models.TextCNN import CNNModel
from models.bpmll import BPMLLModel
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
        elif conf.model_type == 'CNN':
            model = CNNModel(conf.filter_sizes, conf.num_filters, num_words,
                             conf.embedding_size, conf.sequence_length, optimizer, conf.num_classes)
        elif conf.model_type == 'BPMLL':
            model = BPMLLModel(optimizer, conf.num_classes)
        metric = Metrics()
        print("开始训练 ....")
        model.fit(x_train, y_train, batch_size=conf.batch_size, epochs=conf.epochs,
                  callbacks=[metric, TensorBoard(log_dir='log/'+conf.model_type+'/')],
                  validation_data=[x_dev, y_dev], shuffle=True)
        model.summary()
        with open('log/' + conf.model_type + '/' +
                          time.strftime('_%Y_%m_%d', time.localtime(time.time()))+'.json',
                  'w', encoding='utf-8') as ff:
            indices = {'f1': [metric.val_f1s_macro, metric.val_f1s_micro],
                   'precision': [metric.val_precisions_macro, metric.val_precisions_micro],
                   'recall': [metric.val_recalls_macro, metric.val_precisions_micro]}
            json.dump(indices, ff)


