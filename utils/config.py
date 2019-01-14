class configure():
    def __init__(self):
        self.train_path = 'data/sample20000_train.txt'
        self.test_path = 'data/sample20000_test.txt'
        self.cut_path = "data/sample/cutted_text_"
        # 文本的最大词数
        self.sequence_length = 1000
        # 字典的最大长度
        self.max_words = 20000
        # 词向量的维度
        self.embedding_size = 300
        self.num_classes = 202
        self.batch_size = 64
        self.epochs = 100
        self.log_path = 'log/'
        self.libsvm_train = 'data/sample/train.libsvm'
        self.libsvm_test = 'data/sample/test.libsvm'
        self.libsvm_path = self.libsvm_test
        self.model_type = 'BPMLL'
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 128


conf = configure()
