import os
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from operator import itemgetter
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from utils.LoadData import preprocess
from utils.config import conf


def loss_fun(y, y_pre):
    shape = tf.shape(y)
    y_i = tf.equal(y, tf.ones(shape))
    y_not_i = tf.equal(y, tf.zeros(shape))

    # get indices to check
    truth_matrix = tf.to_float(pairwise_and(y_i, y_not_i))

    # calculate all exp'd differences
    # through and with truth_matrix, we can get all c_i - c_k(appear in the paper)
    sub_matrix = pairwise_sub(y_pre, y_pre)
    exp_matrix = tf.exp(tf.negative(sub_matrix))

    # check which differences to consider and sum them
    sparse_matrix = tf.multiply(exp_matrix, truth_matrix)
    sums = tf.reduce_sum(sparse_matrix, axis=[1, 2])

    # get normalizing terms and apply them
    y_i_sizes = tf.reduce_sum(tf.to_float(y_i), axis=1)
    y_i_bar_sizes = tf.reduce_sum(tf.to_float(y_not_i), axis=1)
    normalizers = tf.multiply(y_i_sizes, y_i_bar_sizes)
    loss = tf.divide(sums, normalizers)

    return loss


def pairwise_sub(a, b):
    column = tf.expand_dims(a, 2)
    row = tf.expand_dims(b, 1)
    return tf.subtract(column, row)


def pairwise_and(a, b):
    column = tf.expand_dims(a, 2)
    row = tf.expand_dims(b, 1)
    return tf.logical_and(column, row)


def train_threshold(data_x, data_y, pred):
    data_num = data_x.shape[0]
    label_num = data_y.shape[1]
    threshold = np.zeros([data_num])

    for i in range(data_num):
        pred_i = pred[i, :]
        x_i = data_x[i, :]
        y_i = data_y[i, :]
        tup_list = []
        for j in range(len(pred_i)):
            tup_list.append((pred_i[j], y_i[j]))

        tup_list = sorted(tup_list, key=itemgetter(0))
        min_val = label_num
        for j in range(len(tup_list) - 1):
            val_measure = 0

            for k in range(j + 1):
                if tup_list[k][1] == 1:
                    val_measure = val_measure + 1
            for k in range(j + 1, len(tup_list)):
                if tup_list[k][1] == 0:
                    val_measure = val_measure + 1

            if val_measure < min_val:
                min_val = val_measure
                threshold[i] = (tup_list[j][0] + tup_list[j + 1][0]) / 2

    linreg = LinearRegression()
    linreg.fit(pred, threshold)
    joblib.dump(linreg, "./sk_model/linear_model.pkl")


def estimate(val_predict, val_target):
    _val_f1_macro = f1_score(val_target, val_predict, average='macro')
    _val_f1_micro = f1_score(val_target, val_predict, average='micro')
    _f1 = (_val_f1_macro, _val_f1_micro)
    _val_recall_macro = recall_score(val_target, val_predict, average='macro')
    _val_recall_micro = recall_score(val_target, val_predict, average='micro')
    _recall = (_val_recall_macro, _val_recall_micro)
    _val_precision_macro = precision_score(val_target, val_predict, average='macro')
    _val_precision_micro = precision_score(val_target, val_predict, average='micro')
    _precision = (_val_precision_macro, _val_precision_micro)
    return _f1, _recall, _precision


def BPMLLModel(data_x, data_y, dev_x, dev_y):

    data_num = data_x.shape[0]
    feature_num = data_x.shape[1]
    label_num = data_y.shape[1]
    hidden_unit = int(feature_num * 0.8)
    print("data_num:{0}\nfeature_num:{1}\nlabel_num{2}\nhidden_unit{3}".format(data_num,feature_num,label_num,hidden_unit))
    alpha = 0.1
    batch_size = 32
    epochs = 100
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    max_f1 = 0
    with sess.as_default():
        x = tf.placeholder(tf.float32, [None, feature_num], name='input_x')
        y = tf.placeholder(tf.float32, [None, label_num], name='input_y')

        w1 = tf.Variable(tf.random_normal([feature_num, hidden_unit], stddev=1, seed=1))
        w2 = tf.Variable(tf.random_normal([hidden_unit, label_num], stddev=1, seed=1))

        bias1 = tf.Variable(tf.random_normal([hidden_unit], stddev=0.01, seed=1))
        bias2 = tf.Variable(tf.random_normal([label_num], stddev=0.01, seed=1))

        a = tf.nn.tanh(tf.matmul(x, w1) + bias1)
        pred = tf.nn.tanh(tf.matmul(a, w2) + bias2)
        tf.add_to_collection('pred_network', pred)
        loss = loss_fun(y, pred) + tf.contrib.layers.l2_regularizer(alpha)(w1) + tf.contrib.layers.l2_regularizer(alpha)(w2)
        avg_loss = tf.reduce_mean(loss)
        optimazer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

        init = tf.global_variables_initializer()
        sess.run(init)
        steps = int(data_num / batch_size)
        for epoch in range(epochs):
            epoch_loss = 0
            cnt = 0
            for i in range(steps):
                start = i*batch_size
                end = min(start+batch_size, data_num)
                loss_summary = tf.summary.scalar("loss", avg_loss)
                feed_dic = {x: data_x[start:end],
                            y: data_y[start:end]}
                _,__, los = sess.run([optimazer, loss_summary, avg_loss], feed_dict=feed_dic)
                if i%100==0:
                    print("epoch: {0}  step:{1}  loss:{2}".format(epoch, i, los))
                cnt = cnt + 1
                epoch_loss = epoch_loss + los
            epoch_loss = epoch_loss/cnt
            y_pred = sess.run(pred, feed_dict={x: dev_x})
            print("y_pred shape:", len(y_pred), len(y_pred[0]))
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0
            _f1, _recall, _precision = estimate(y_pred, dev_y)
            del y_pred
            print("EPOCH:", epoch)
            print("loss:{0}\n".format(epoch_loss))
            print("MACRO:  --F1:{0}     --Recall:{1}    --Precision:{2}".format(_f1[0], _recall[0], _precision[0]))
            print("MICRO:  --F1:{0}     --Recall:{1}    --Precision:{2}".format(_f1[1], _recall[1], _precision[1]))
            print("-------------------------------------------------------------------------------")
            with open("log/bpmll/BPMLL_LOG.txt", 'a', encoding='utf-8') as f:
                f.write("EPOCH:{0}\n\r".format(epoch))
                f.write("loss:{0}\n\r".format(epoch_loss))
                f.write("MACRO:  --F1:{0}     --Recall:{1}    --Precision:{2}".format(_f1[0], _recall[0], _precision[0]))
                f.write("MICRO:  --F1:{0}     --Recall:{1}    --Precision:{2}".format(_f1[1], _recall[1], _precision[1]))
                f.write("**********************************************************************")
            if _f1[0] > max_f1:
                max_f1 = _f1[0]
                predict = sess.run(pred, feed_dict={x: data_x})
                train_threshold(data_x, data_y, predict)
                saver = tf.train.Saver()
                saver.save(sess, "./tf_model/{0}".format(max_f1))


def test(x_test, y_test, model_path):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./tf_model/model.meta')
        saver.restore(sess, 'tf_model/model')
        graph = tf.get_default_graph()
        pred = tf.get_collection('pred_network')[0]
        x = graph.get_operation_by_name('input_x').outputs[0]

        pred = sess.run(pred, feed_dict={x: x_test})

    linreg = joblib.load('./sk_model/linear_model.pkl')
    threshold = linreg.predict(pred)
    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0
    _f1, _recall, _precision = estimate(pred, y_test)
    del pred
    print("MACRO:  --F1:{0}     --Recall:{1}    --Precision:{2}".format(_f1[0], _recall[0], _precision[0]))
    print("MICRO:  --F1:{0}     --Recall:{1}    --Precision:{2}".format(_f1[1], _recall[1], _precision[1]))
    print("-------------------------------------------------------------------------------")
    with open("log/bpmll/BPMLL_test_LOG.txt", 'a', encoding='utf-8') as f:
        f.write(model_path+" : \n\r")
        f.write("MACRO:  --F1:{0}     --Recall:{1}    --Precision:{2}".format(_f1[0], _recall[0], _precision[0]))
        f.write("MICRO:  --F1:{0}     --Recall:{1}    --Precision:{2}".format(_f1[1], _recall[1], _precision[1]))
        f.write("**********************************************************************")


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test, word_index = preprocess(conf.train_path, conf.test_path)
    np.savez('../data/np_data.npz', X_train, Y_train, X_test, Y_test)
    BPMLLModel(X_train, Y_train, X_test, Y_test)




