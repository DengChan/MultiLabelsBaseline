import numpy as np
import time
from sklearn.externals import joblib
from utils.LoadData import ml_load_data
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import GridSearchCV
from utils.Estimate import ml_estimate

def find_best_k(x_train, y_train):
    parameters = {'k': range(20, 50), 's': [1.0]}
    clf = GridSearchCV(MLkNN(), parameters, scoring='f1_macro')
    clf.fit(x_train, y_train)
    besk_k = clf.best_params_[0]['k']
    print(clf.best_params_, clf.best_score_)
    '''
    输出的例子
    ({'k': 1, 's': 0.5}, 0.45223607257008969)
    '''
    return besk_k


def mlknn(k=20):

    parameters = {'k': range(1, 3), 's': [0.5, 0.7, 1.0]}
    print("running ML-KNN ...")
    X_train, Y_train, X_test, Y_test = ml_load_data()
    print("Loading data done\n Start training ..")
    clf = MLkNN(k=k)
    clf.fit(X_train, Y_train)
    print("training done\nStart testing...")
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

    # 保存模型
    print("saving model...")
    model_name = 'f1_' + str(_f1[0]) + time.strftime('_%Y_%m_%d', time.localtime(time.time()))
    joblib.dump(clf, "checkpoints/mlknn/{0}.m".format(model_name))
    # 读取 clf= joblib.load(path)
    print('MACRO: — val_f1: %f — val_precision: %f — val_recall %f' % (
        _f1[0], _precision[0], _recall[0]))
    print('MICRO: — val_f1: %f — val_precision: %f — val_recall %f' % (
        _f1[1], _precision[1], _recall[1]))

    with open('log/mlknn/' + model_name + '.txt', 'w', encoding='utf-8') as f:
        f.write('MACRO: \n— val_f1: %f \n— val_precision: %f \n— val_recall %f\n' % (
            _f1[0], _precision[0], _recall[0]))
        f.write('MICRO: \n— val_f1: %f \n— val_precision: %f \n— val_recall %f\n' % (
            _f1[1], _precision[1], _recall[1]))

    print("------------------ ML-KNN 计算结束---------------------")





