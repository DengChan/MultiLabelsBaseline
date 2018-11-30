
class configure():
    def __init__(self):
        self.train_path = '../data/data_train.json'
        self.test_path = '../data/data_valid.json'
        # 文本的最大词数
        self.sequence_length = 1000
        # 字典的最大长度
        self.max_words = 20000
        # 词向量的维度
        self.embedding_size = 300
        self.dataset_path = '../data/CAIL_dataset.json'
        self.libsvm_path = '../data/CAIL_test.libsvm'

conf = configure()