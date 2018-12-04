import numpy as np
import time
from sklearn.externals import joblib
from utils.LoadData import ml_load_data
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from utils.Estimate import ml_estimate


def CC():
    print("running CC-GaussianNB ...")
    X_train, Y_train, X_test, Y_test = ml_load_data()
    print("Loading data done\n Start training ..")

    clf = ClassifierChain(GaussianNB())

    clf.fit(X_train, Y_train)
    print("training done")

    # 保存模型
    print("saving model...")
    model_name = 'f1_' + str(_f1[0]) + time.strftime('_%Y_%m_%d', time.localtime(time.time()))
    joblib.dump(clf, "checkpoints/CC-GaussianNB/{0}.m".format(model_name))
    # 读取 clf= joblib.load(path)

    print("start predicting ...")
    # 预测值
    val_predict = np.asarray(clf.fit(X_test))
    # 目标值
    val_target = Y_test
    print("*********************预测值*****************************")
    print("预测y长度: ", len(val_predict))
    print(val_predict[0:5])
    print("*********************目标值*****************************")
    print("目标y长度: ", len(val_target))
    print(val_target[0:5])
    print("********************************************************")

    print("计算指标 ...")
    # [0]:macro [1]:micro
    _f1, _recall, _precision = ml_estimate(val_predict, val_target)


    print('MACRO: — val_f1: %f — val_precision: %f — val_recall %f' % (
        _f1[0], _precision[0], _recall[0]))
    print('MICRO: — val_f1: %f — val_precision: %f — val_recall %f' % (
        _f1[1], _precision[1], _recall[1]))

    with open('log/CC-GaussianNB/' + model_name + '.txt', 'w', encoding='utf-8') as f:
        f.write('MACRO: \n— val_f1: %f \n— val_precision: %f \n— val_recall %f\n' % (
            _f1[0], _precision[0], _recall[0]))
        f.write('MICRO: \n— val_f1: %f \n— val_precision: %f \n— val_recall %f\n' % (
            _f1[1], _precision[1], _recall[1]))

    print("------------------ CC-GaussianNB 计算结束---------------------")