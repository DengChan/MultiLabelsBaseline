import numpy
from utils.CalTFIDF import cal_TF_IDF
from utils.CalTFIDF import cal_TF_IDF_2
from utils.config import conf


'''
把数据转为.libsvm格式
x_array:词袋模型表示的文本集，单个文本长度为1000，总词数20000
y_array:文本对应的标签列表，可能包含一个或多个标签的index
train_len: 训练集的长度

生成数据格式 
忽略[ ]
[label1,label2,label3...] [word_index:tfidf] [word_index:tfidf] [word_index:tfidf] .......

'''


def to_libsvm(x_array, y_array, train_len):
    length = len(x_array)
    print("calculating TF-IDF ....")
    x_tf_idf = cal_TF_IDF(x_array)
    print("Done with calculating TF-IDF.")

    print("Converting nparray to libsvm ...")
    page = -1
    for i in range(length):
        line = ""
        first = True

        for j in range(len(y_array[i])):
            if first:
                line = line+str(y_array[i][j])
                first = False
            else:
                line = line+','+str(y_array[i][j])
        # 取出该文档中所有的单词及其TF-IDF 并按序排列
        tmp = {}
        for j in range(len(x_array[i])):
            id = int(x_array[i][j])
            value = x_tf_idf[i][j]
            # 小于1e-9的认为该值为0跳过，如果存在该词 跳过
            if ((value-0.0) < 1e-9) or (id in tmp):
                continue
            tmp[id] = value
        # 字典按key值排序，返回元祖的列表
        tmp = sorted(tmp.items(), key=lambda tmp:tmp[0])
        # 按顺序和格式写入
        for k, v in tmp:
            line = line + ' {0}:{1}'.format(str(k), str(v))
        line = line+'\n'
        # 训练集写入
        if i < train_len:
            with open(conf.libsvm_train, 'a', encoding='utf-8') as f:
                f.write(line)
        else:
            # 测试集写入
            with open(conf.libsvm_test, 'a', encoding='utf-8') as f:
                f.write(line)

        # print("example of libsvm:\n"+line)
    print("Converting work done. ")


def to_libsvm2(x_array, y_array, train_len):
    length = len(x_array)
    print("calculating TF-IDF ....")
    x_tf_idf = cal_TF_IDF_2(x_array)
    print("Done with calculating TF-IDF.")

    print("Converting nparray to libsvm ...")
    for i in range(length):
        line = ""
        first = True
        for j in range(len(y_array[i])):
            if first:
                line = line + str(y_array[i][j])
                first = False
            else:
                line = line + ',' + str(y_array[i][j])

        for j in range(len(x_tf_idf[i])):
            value = x_tf_idf[i][j]
            # 小于1的认为该值为0跳过，如果存在该词 跳过
            if (value - 0.0) < 1e-9:
                continue
            else:
                line = line + ' {0}:{1}'.format(j+1, value)
        line = line+'\n'
        if i < train_len:
            with open(conf.libsvm_train, 'a', encoding='utf-8') as f:
                f.write(line)
        else:
            # 测试集写入
            with open(conf.libsvm_test, 'a', encoding='utf-8') as f:
                f.write(line)
    print("Converting work done. ")








