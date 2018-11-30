import os
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from utils.config import conf

def svm():
    print("running SVM ...")
    train_path = '../data/CAIL_train.libsvm'
    test_path = '../data/CAIL_test.libsvm'
    # 读取数据
    X_train, Y_train = load_svmlight_file(train_path, n_features=conf.sequence_length, dtype=np.float64, multilabel=True)
    print("SHAPE: X:{0}    y:{1}".format(X_train.toarray.shape(), Y_train.shape()))
    # 打乱数据
    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_test, Y_test = load_svmlight_file(test_path, n_features=conf.sequence_length, dtype=np.float64, multilabel=True)
    # 把label转为one-hot
    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(Y_train)
    Y_test = mlb.fit_transform(Y_test)
    # 训练
    clf = OneVsRestClassifier(SVC(kernel='linear', verbose=True), n_jobs=-1)
    clf.fit(X_train, Y_train)
    # 测试
    val_predict = np.asarray(clf.fit(X_test))
    val_target = Y_test
    print("*********************预测值*****************************")
    print("预测y长度: ", len(val_predict))
    print(val_predict[0:5])
    print("*********************目标值*****************************")
    print("目标y长度: ", len(val_target))
    print(val_target[0:5])
    print("********************************************************")
    print("计算指标 ...")
    _val_f1_macro = f1_score(val_target, val_predict, average='macro')
    _val_f1_micro = f1_score(val_target, val_predict, average='micro')
    _val_recall_macro = recall_score(val_target, val_predict, average='macro')
    _val_recall_micro = recall_score(val_target, val_predict, average='micro')
    _val_precision_macro = precision_score(val_target, val_predict, average='macro')
    _val_precision_micro = precision_score(val_target, val_predict, average='micro')

    # 保存模型
    joblib.dump(clf, "checkpoints/svm_f1_{0}.m".format(_val_f1_macro))
    # 读取 clf= joblib.load(path)
    print('MACRO: — val_f1: %f — val_precision: %f — val_recall %f' % (
        _val_f1_macro, _val_precision_macro, _val_recall_macro))
    print('MICRO: — val_f1: %f — val_precision: %f — val_recall %f' % (
        _val_f1_micro, _val_precision_micro, _val_recall_micro))
    print("--------------SVM计算结束---------------")

svm()


