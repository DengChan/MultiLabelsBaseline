import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from skmultilearn.problem_transform import ClassifierChain
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
import time
from skmultilearn.adapt import MLkNN
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer
from utils.config import conf
from utils.LoadData import load_data
from utils.ToLibsvm import to_libsvm2
from utils.ToLibsvm import to_libsvm
import scipy
import os


k = 20

max_words = 20000 # 字典的大小

epochs = 100

num_classes=130
namedic = {0: 'svm', 1: 'MLKNN', 2: 'CC',3:'SGD_SVM'}

mlb = MultiLabelBinarizer()


def GetLibsvm(train_path, test_path):
    """
        预处理数据并读取
        :param train_path:
        :param test_path:
        :return: 训练集x,y 测试集x,y
        """

    # accu_dic是存放的已有罪名的字典 罪名对应序号
    accu_dic = {}
    # acc_cnt是已读取的不同罪名数
    acc_cnt = 0
    train_texts, train_labels, acc_cnt = load_data(train_path, 0, accu_dic, acc_cnt)
    print('Load Train data Done')
    print("data len: {0},dic cnt: {1}".format(len(train_texts), acc_cnt))
    test_texts, test_labels, acc_cnt = load_data(test_path, 1, accu_dic, acc_cnt)
    print('Load Text data Done')
    print("data len: {0},dic cnt: {1}".format(len(test_texts), acc_cnt))
    # train_texts = np.array(train_texts)
    # test_texts = np.array(test_texts)

    # 合并
    train_len = len(train_texts)
    texts = train_texts + test_texts
    old_labels = train_labels + test_labels
    '''这里转为libsvm,输入单个x 分词后的句子,单个y为罪名的列表'''
    to_libsvm2(texts, old_labels, train_len)


def ml_load_data(train_path='data/CAIL_train.libsvm', test_path='data/CAIL_test.libsvm'):

    # 读取数据
    print("Loading data ...")
    X_train, Y_train = load_svmlight_file(train_path, n_features=max_words, dtype=np.float64, multilabel=True)
    # X_exam = X_train.toarray()
    # del X_exam
    # 打乱数据
    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_test, Y_test = load_svmlight_file(test_path, n_features=max_words, dtype=np.float64, multilabel=True)
    # 把label转为one-hot
    Y_train = mlb.fit_transform(Y_train)
    Y_test = mlb.transform(Y_test)
    Y_train = scipy.sparse.csr_matrix(Y_train)
    Y_test = scipy.sparse.csr_matrix(Y_test)
    # print("mlb classes:\n", len(mlb.classes_))
    # print(mlb.classes_)
    print('X_train shape:',X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('X_test shape:',X_test.shape)
    print('Y_test shape:',Y_test.shape)
    return X_train, Y_train, X_test, Y_test


def ml_estimate(val_predict, val_target):
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


def data_iter():
    for i in range(553):
        X_train, Y_train = load_svmlight_file('data/hls/steps/hls_train_step_{0}.libsvm'.format(i), n_features=max_words, dtype=np.float64, multilabel=True)
        X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
        X_train = X_train.toarray()
        Y_train = mlb.transform(Y_train)
        Y_train = to_one_label(Y_train)
        yield X_train, Y_train

# 主程序
def ml(X_train, Y_train, X_test, Y_test, type=0, model_path='None', train_mode=0):

    print("running {0} ...".format(namedic[type]))
    print(" Start training ..")
    # 训练 默认为SGD的SVM
    clf = SGDClassifier(loss='hinge', penalty='l2')
    if type == 1 and model_path == 'None':
        clf = MLkNN(k=10, s=0.5)
    # CC 但容易爆内存 未使用
    elif type == 2 and model_path == 'None':
        clf = ClassifierChain(SVC())
    # 如果路径不为空，则读取模型
    elif model_path != 'None':
        clf = joblib.load(model_path)

    if train_mode == 0:
        clf.fit(X_train, Y_train)
        del X_train, Y_train
    else:
        # 增量学习 单标签模式
        del X_train, Y_train
        for ep in range(epochs):
            data_it = data_iter()
            for i, (X, Y) in enumerate(data_it):
                # 得到标签号
                Y = np.asarray(np.argmax(Y, 1),dtype=np.int32)
                # print("Y :", Y.shape)

                clf.partial_fit(X, Y, classes=np.array(range(num_classes), dtype=np.int32))
            val_predict = np.asarray(clf.predict(X_test), dtype=np.int32)
            _f1, _recall, _precision = ml_estimate(val_predict, Y_test)
            print("EPOCH:{0}".format(ep))
            print('MACRO: — val_f1: %f — val_precision: %f — val_recall %f' % (
                _f1[0], _precision[0], _recall[0]))
            print('MICRO: — val_f1: %f — val_precision: %f — val_recall %f' % (
                _f1[1], _precision[1], _recall[1]))

    print("training done\nStart testing...")
    # 测试
    val_predict = clf.predict(X_test)
    # print(type(val_predict))
    # print(type(Y_test))
    print(val_predict.shape)
    print(Y_test.shape)

    print("计算指标 ...")
    # [0]:macro [1]:micro
    _f1, _recall, _precision = ml_estimate(val_predict, Y_test)

    # 保存模型
    print("saving model...")
    model_name = 'f1_' + str(_f1[0]) + time.strftime('_%Y_%m_%d', time.localtime(time.time()))
    joblib.dump(clf, "checkpoints/{0}/{1}.m".format(namedic[type], model_name))
    # 读取 clf= joblib.load(path)
    print('MACRO: — val_f1: %f — val_precision: %f — val_recall %f' % (
        _f1[0], _precision[0], _recall[0]))
    print('MICRO: — val_f1: %f — val_precision: %f — val_recall %f' % (
        _f1[1], _precision[1], _recall[1]))

    with open('log/'+namedic[type]+'/'+'f1_' + str(_f1[0])+time.strftime('_%Y_%m_%d', time.localtime(time.time()))+'.txt', 'w', encoding='utf-8') as f:
        f.write('MACRO: \n\r— val_f1: %f \n\r— val_precision: %f \n\r— val_recall %f\n\r' % (
        _f1[0], _precision[0], _recall[0]))
        f.write('MICRO: \n\r— val_f1: %f \n\r— val_precision: %f \n\r— val_recall %f\n\r' % (
        _f1[1], _precision[1], _recall[1]))

    print("--------------{0}计算结束---------------".format(namedic[type]))


def to_one_label(y):
    for i in range(len(y)):
        yi = y[i]
        index = np.argmax(yi)
        yi_tmp = np.zeros_like(yi, dtype=np.int32)
        yi_tmp[index] = 1
        y[i] = yi
    return y


if __name__ == '__main__':
    # 增量学习好像不是这么写的 先不用

    if not os.path.exists(conf.libsvm_train):
        GetLibsvm(conf.train_path, conf.test_path)

    X_train, Y_train, X_test, Y_test = ml_load_data(conf.libsvm_train, conf.libsvm_test)

    # 如果单标签就把这三行注释去掉
    # Y_train = to_one_label(Y_train)
    # Y_test = to_one_label(Y_test)
    # Y_test = Y_test.todense()

    # type = 0 :SVM  1:MLKNN 2:CC
    ml(X_train, Y_train, X_test, Y_test, type=1, train_mode=0)
