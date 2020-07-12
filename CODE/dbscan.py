import os
import re
import tqdm
import scipy
import pickle
import numpy as np
import pandas as pd
from jieba import lcut
from hanziconv import HanziConv
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from keras.preprocessing.text import Tokenizer
from sklearn.decomposition import TruncatedSVD
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from model import *
from data_loader import *

#DBSCAN的参数其实就两个，一个是半径Eps，一个是半径范围内的最少点数量MinPts

def read_train_data(tests):
    X_train_list = []
    X_train_time_list = []

    sum = 0

    for d in tests:
        # story_id,event_id,content,keywords,main_keywords,time
        X_train_list.append(d.keywords)
        X_train_time_list.append(d.time)

    corpus = []
    for i, text in enumerate(X_train_list):
        text = text.split(' ')
        corpus.append(TaggedDocument(text, tags=[i]))

    doc2vec_model = Doc2Vec(corpus, min_count=1, window=50, size=100,
                            sample=1e-5, workers=2)

    doc2vec_model.train(corpus,
                        total_examples=doc2vec_model.corpus_count,
                        epochs=5)
    encoder_data = doc2vec_model.docvecs.doctag_syn0

    print(encoder_data)
    return encoder_data
    #return X_train_list
    #return X_train_list
    # for i in range(len(X_train_list)):
    #     tokenizer = Tokenizer()
    #     tokenizer.fit_on_texts(X_train_list[i])
    #     # tokenizer.word_index
    #     sum += len(tokenizer.word_index)
    #
    #     X_train_list[i] = tokenizer.texts_to_sequences(X_train_list[i])
    #     X_train_list[i] = sequence.pad_sequences(X_train_list[i], maxlen=200)
    #
    # print('tokens的个数：', sum)

def doc2vec(cut_text_list):
    """
    采用doc2vec将文本进行向量化
    :param cut_text_list: 分词后的文本列表，词汇之间采用空格连接. [list]
    :param min_count: 最低词频. [int]
    :param window: doc2vec窗口大小. [int]
    :param size: doc2vec向量的维度大小. [int]
    :param sample: 高频词负采样的概率. [float]
    :param negative: 负采样的词汇数. [int]
    :param workers: 进程数. [int]
    :return:
    """
    corpus = []
    for i, text in enumerate(cut_text_list):
        text = text.split(' ')
        corpus.append(TaggedDocument(text, tags=[i]))

    doc2vec_model = Doc2Vec(corpus, min_count=5, window=50, size=100,
                            sample=1e-5,  workers=2)

    doc2vec_model.train(corpus,
                        total_examples=doc2vec_model.corpus_count,
                        epochs=5)
    encoder_data = doc2vec_model.docvecs.doctag_syn0

    return encoder_data

def svd(encoder_data, dim=2):
    """
    对向量化后的数据进行svd降维
    :param encoder_data: 向量化后的数据. [np.array or csr_matrix]
    :param dim: 降维后的维度大小. [int]
    :return:
    """
    svd = TruncatedSVD(dim)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    encoder_data = lsa.fit_transform(encoder_data)
    return encoder_data

def dbscan(encoder_data):
    db = DBSCAN(eps=0.001, min_samples=10).fit(encoder_data) #若样本在数据集中存在eps距离内有至少min_samples
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # C_min
    C_miss = 0.5
    C_FA = 0.5
    P_miss = 0
    P_FA = 0
    tags = {}
    for d in labels:
        if d not in tags:
            tags[d] = 0
        tags[d] += 1
    tcount = list(sorted([count for count in tags.values()], reverse=True))[0]
    tid = list(tags.keys())[0]
    fa = 0
    miss = 0

    for d in labels:
        if d != tid:
            fa += 1

    for d in labels:
        if d == tid:
            miss += 1

    miss = miss / (miss + tcount)
    P_FA += fa
    P_miss += miss

    # print 'P_FA',P_FA/len(newClusters)
    # print 'P_Miss',P_miss/len(newClusters)
    print('C_min', C_FA * P_FA / len(tags) + C_miss * P_miss / len(tags))


# def plot_cluster(encoder_data):
#     """
#     绘制聚类后的效果图
#     :param encoder_data: 向量表示后的数据，可以是np.array或者csr_matrix的形式
#     :param label: 聚类后的标签
#     :param reduction_method: 降维的方法，可以是pca或者svd
#     :return:
#     """
#     plt.switch_backend('agg')
#     plt.figure()
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False
#     encoder_data = svd(encoder_data, dim=2)
#     label = dbscan(encoder_data)
#     data = pd.DataFrame()
#     data['x1'] = encoder_data[:, 0]
#     data['x2'] = encoder_data[:, 1]
#     data['label'] = label
#     unique_label = list(set(label))
#     colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_label))]
#     for col, lab in zip(colors, unique_label):
#         if lab == -1:
#             col = [0, 0, 0, 1]
#             continue
#         this_cluster = data[data.label == lab]
#         plt.scatter(this_cluster.x1, this_cluster.x2, c=tuple(col), s=5)
#     plt.title('总共聚类的数目为：{0}'.format(len(unique_label) - 1))
#     plt.savefig('cluster.png')



LEN_OF_TRAIN= 200 #20000 #200

data = getData() # docuemnt集合

# document的数量： 25941
trains = data[:LEN_OF_TRAIN]

tests = data[LEN_OF_TRAIN:20000]

encoder_data=read_train_data(tests)

