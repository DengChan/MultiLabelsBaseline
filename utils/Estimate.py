from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import tensorflow.keras as kr
import numpy as np
import time
from utils.config import conf


class Metrics(kr.callbacks.Callback):
    def __init__(self):
        self.val_f1s_macro = []
        self.val_recalls_macro = []
        self.val_precisions_macro = []
        self.val_f1s_micro = []
        self.val_recalls_micro = []
        self.val_precisions_micro = []
        # def on_train_begin(self, logs={}):
        #   self.val_f1s = []
        #   self.val_recalls = []
        #  self.val_precisions = []

    def on_train_begin(self, logs=None):
        self.val_f1s_macro = []
        self.val_recalls_macro = []
        self.val_precisions_macro = []
        self.val_f1s_micro = []
        self.val_recalls_micro = []
        self.val_precisions_micro = []

    def on_epoch_end(self, epoch, logs={}):
        # 默认predict是batch=32
        val_predict = np.asarray(self.model.predict(self.validation_data[0], verbose=0))
        val_predict[val_predict > 0.5] = 1
        val_predict[val_predict <= 0.5] = 0
        print("*********************预测值*****************************")
        print("预测y长度: ", len(val_predict))
        print(val_predict[0:5])
        val_target = self.validation_data[1]
        print("*********************目标值*****************************")
        print("目标y长度: ", len(val_target))
        print(val_target[0:5])

        _val_f1_macro = f1_score(val_target, val_predict, average='macro')
        _val_f1_micro = f1_score(val_target, val_predict, average='micro')
        _val_recall_macro = recall_score(val_target, val_predict, average='macro')
        _val_recall_micro = recall_score(val_target, val_predict, average='micro')
        _val_precision_macro = precision_score(val_target, val_predict, average='macro')
        _val_precision_micro = precision_score(val_target, val_predict, average='micro')

        # _val_accuracy = accuracy_score(val_target, val_predict)
        self.val_f1s_macro.append(_val_f1_macro)
        self.val_recalls_macro.append(_val_recall_macro)
        self.val_precisions_macro.append(_val_precision_macro)
        self.val_f1s_micro.append(_val_f1_micro)
        self.val_recalls_micro.append(_val_recall_micro)
        self.val_precisions_micro.append(_val_precision_micro)
        model_name = conf.model_type+'f1_'+str(_val_f1_macro)+time.strftime('_%Y_%m_%d', time.localtime(time.time()))
        self.model.save('checkpoints/'+model_name+'.h5')
        # self.model.summary()
        print('MACRO: — val_f1: %f — val_precision: %f — val_recall %f' % (
            _val_f1_macro, _val_precision_macro, _val_recall_macro))
        print('MICRO: — val_f1: %f — val_precision: %f — val_recall %f' % (
            _val_f1_micro, _val_precision_micro, _val_recall_micro))
        with open('log/'+conf.model_type+time.strftime('_%Y_%m_%d', time.localtime(time.time()))+'.txt', 'a', encoding='utf-8') as ff:
            ff.write('EPOCH: {0}\n'.format(epoch))
            ff.write('MACRO: — val_f1: %f — val_precision: %f — val_recall %f\n' % (
             _val_f1_macro, _val_precision_macro, _val_recall_macro))
            ff.write('MICRO: — val_f1: %f — val_precision: %f — val_recall %f\n' % (
             _val_f1_micro, _val_precision_micro, _val_recall_micro))
            ff.write('----------------------------------------------------------------------\n')
        # print('— 验证集的F1-SCORE: %f ' % _val_f1)
        # print("-----------------------------模型已保存------------------------")
        return


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

def test():
    l1 =np.asarray([
          [1,0,0],
          [0,1,1],
          [0,0,1],
          [1,0,1]],dtype=np.int32)
    l2 = np.asarray([
          [1,0,0],
          [0,1,0],
          [0,0,1],
          [1,0,0]],dtype=np.int32)
    f1 = f1_score(l1, l2, average='macro')
    print(f1)
    f1 = f1_score(l1, l2, average='micro')
    print(f1)
    p = precision_score(l1, l2, average='macro')
    print(p)

# test()

