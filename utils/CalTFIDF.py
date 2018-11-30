from sklearn.feature_extraction.text import TfidfTransformer
import numpy


def cal_TF_IDF(texts_sequences):
    x = texts_sequences
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(x)
    tfidf = tfidf.toarray()
    return tfidf