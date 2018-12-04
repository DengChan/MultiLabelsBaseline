import os
import json
import jieba
import codecs
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow.keras as kr
from utils.config import conf
from utils.ToLibsvm import to_libsvm
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle


def load_data(path, data_type, accu_dic, acc_cnt):

    '''
    :param path:数据路径
    :param data_type: 0，1。 0表示训练数据，1表示测试数据
    :param accu_dic: 罪名索引字典
    :param acc_cnt: 罪名数
    :return: 按顺序存放的text,罪名,罪名数
    '''

    print("reading data..")
    fin = open(path, 'r', encoding='utf8')
    all_text = []
    line = fin.readline()
    # 按顺序存放对应的罪名
    accu_label = []
    while line:
        # 一条数据是一行 按行读取
        d = json.loads(line)
        all_text.append(d['fact'].strip().replace("\n", ""))
        # label_list存放一条数据的所有罪名编号
        label_list = []

        for lb in d['meta']['accusation']:
            lb = lb.strip()
            if not (lb in accu_dic):
                accu_dic[lb] = acc_cnt
                label_list.append(acc_cnt)
                acc_cnt = acc_cnt+1
            else:
                label_list.append(accu_dic[lb])
        accu_label.append(label_list)
        line = fin.readline()
    fin.close()

    # 处理y 序列化
    print("共有{0}个罪名".format(acc_cnt))

    # 处理输入文本
    loaded_text = []
    if not os.path.exists(conf.cut_path+str(data_type)+".txt"):
        print("cut text...")
        cnt = 0

        for text in all_text:
            cnt = cnt+1
            text = text.strip()
            cutted = jieba.lcut(text)
            # 以空格间隔
            cutted = " ".join(cutted)
            print("{0}  :  {1}".format(cnt, cutted))
            loaded_text.append(cutted)

        # 写入文件
        f = codecs.open(conf.cut_path+str(data_type)+".txt", "w", "utf-8")
        for line in loaded_text:
            f.write(line)
            f.write('\n')
        f.close()
    else:
        with open(conf.cut_path+str(data_type)+".txt", encoding='utf-8') as f:
            loaded_text = f.read().splitlines()
        print("数据的长度为", len(loaded_text))

    return loaded_text, accu_label, acc_cnt


def preprocess(train_path, test_path):

    """
    预处理数据并读取
    :param train_path:
    :param test_path:
    :return: 训练集x,y 测试集x,y
    """
    x_train = []
    x_dev = []
    y_train = []
    y_dev = []
    '''
    if not os.path.exists(dataset_path):
    '''
    # accu_dic是存放的已有罪名的字典 罪名对应序号
    accu_dic = {}
    # acc_cnt是已读取的不同罪名数
    acc_cnt = 0
    train_texts, train_labels, acc_cnt = load_data(train_path, 0, accu_dic, acc_cnt)
    print('Load Train data Done')
    print("dic len: {0},cnt: {1}".format(len(accu_dic), acc_cnt))
    test_texts, test_labels, acc_cnt = load_data(test_path, 1, accu_dic, acc_cnt)
    print('Load Text data Done')
    print("dic len: {0},cnt: {1}".format(len(accu_dic), acc_cnt))
    #train_texts = np.array(train_texts)
    #test_texts = np.array(test_texts)

    '''
    np.random.seed(0)
    shuffle_indices_2 = np.random.permutation(np.arange(len(test_text)))
    test_text = test_text[shuffle_indices_2]
    test_labels = test_labels[shuffle_indices_2]
    '''

    # 合并
    train_len = len(train_texts)
    texts = train_texts+test_texts
    old_labels = train_labels+test_labels



    # 文本的最大词数
    sequence_length = conf.sequence_length
    # 字典的最大长度
    max_words = conf.max_words

    # 单词序列化,对训练数据编码
    tokenizer = kr.preprocessing.text.Tokenizer(num_words=max_words, lower=True)
    # 生成token字典
    tokenizer.fit_on_texts(texts)
    # 转换为向量表示
    sequences = tokenizer.texts_to_sequences(texts)

    # print("sequence example:   ", sequences[0])
    # print(tokenizer.word_counts)
    # print(tokenizer.word_index)
    # print(tokenizer.word_docs)
    # print(tokenizer.index_docs)

    # 一个dict，保存所有word对应的编号id，从1开始
    word_index = tokenizer.word_index
    # 不足填充,将每条文本的长度设置一个固定值
    data = kr.preprocessing.sequence.pad_sequences(sequences, sequence_length, padding='post')
    # print("data emaxple:     ", data[0])

    '''这里转为libsvm,输入单个x为one-hot,单个y为罪名的列表'''
    #if not os.path.exists(conf.libsvm_path):
    to_libsvm(data, old_labels, train_len)


    # 把输出Y one-hot
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(old_labels)

    print("shape of input: ", data.shape)
    print("shape of output: ", labels.shape)
    print("type of input:", str(type(data)))
    # 将数据分为训练和测试集
    # 打乱训练数据
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(train_len))
    # 分离训练 测试集
    #dev_sample_index = -1 * int(0.2 * float(len(labels)))
    x_train = data[:train_len][shuffle_indices]
    x_dev = data[train_len:]
    y_train = labels[:train_len][shuffle_indices]
    y_dev = labels[train_len:]
    print("labels example: \n", y_dev[12])
    del train_texts, train_labels, test_texts, test_labels, texts, labels, data

    '''
    dataset = {"x_train": x_train.tolist(), "y_train": y_train.tolist(), "x_dev": x_dev.tolist(), "y_dev": y_dev.tolist()}
    with open(dataset_path, 'w', encoding='utf-8') as js:
        json.dump(dataset, js)
    
    else:
        with open(dataset_path, 'r', encoding='utf-8') as js:
            data = json.load(js)
            x_train = np.array(data["x_train"])
            x_dev = np.array(data["x_dev"])
            y_train = np.array(data["y_train"])
            y_dev = np.array(data["y_dev"])

    x =1
    '''
    return x_train, y_train, x_dev, y_dev, word_index


def ml_load_data():
    train_path = 'data/CAIL_train.libsvm'
    test_path = 'data/CAIL_test.libsvm'
    # 读取数据
    print("Loading data ...")
    X_train, Y_train = load_svmlight_file(train_path, n_features=conf.max_words, dtype=np.float64, multilabel=True)
    # X_exam = X_train.toarray()
    # del X_exam
    # 打乱数据
    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_test, Y_test = load_svmlight_file(test_path, n_features=conf.max_words, dtype=np.float64, multilabel=True)
    # 把label转为one-hot
    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(Y_train)
    Y_test = mlb.fit_transform(Y_test)
    return X_train, Y_train, X_test, Y_test

def test():
    a, b, c, d,e = preprocess('../data/data_train.json', '../data/data_test.json')
    #print("{0},{1},{2},{3}".format(len(a),len(b),len(c),len(d)))
    #print(a[0])
    #print(c[0])

#test()
