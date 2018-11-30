from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import tensorflow.keras as kr
import numpy

class Metrics(kr.callbacks.Callback):
    def __init__(self):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    #def on_train_begin(self, logs={}):
     #   self.val_f1s = []
     #   self.val_recalls = []
      #  self.val_precisions = []


    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        # 默认predict是batch=32
        val_predict = np.asarray(self.model.predict(self.validation_data[0], verbose=0))
        val_predict[val_predict>0.5]=1
        val_predict[val_predict<=0.5]=0
        print("*********************预测值*****************************")
        print("预测y长度: ", len(val_predict))
        print(val_predict[0:5])
        val_target = self.validation_data[1]
        print("*********************目标值*****************************")
        print("目标y长度: ", len(val_target))
        print(val_target[0:5])

        _val_f1 = f1_score(val_target, val_predict, average='macro')
        _val_recall = recall_score(val_target, val_predict, average='macro')
        _val_precision = precision_score(val_target, val_predict, average='macro')
        #_val_accuracy = accuracy_score(val_target, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        model_name = 'f1_'+str(_val_f1)+'.h5'
        self.model.save('checkpoints/{0}'.format(model_name))
        print('— val_f1: %f — val_precision: %f — val_recall %f' % (_val_f1, _val_precision, _val_recall))
        #print('— 验证集的F1-SCORE: %f ' % _val_f1)
        #print("-----------------------------模型已保存------------------------")
        return