import os
import re
import tqdm
import scipy
import pickle
import numpy as np
import pandas as pd
import jieba
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
from gensim.test.utils import common_corpus
from gensim.models import LdaSeqModel
from collections import defaultdict
from gensim import corpora
from model import *
from data_loader import *
# def read_train_data(tests):
#     X_train_list = []
#     X_train_time_list = []
#
#     sum = 0
#
#     for d in tests:
#         # story_id,event_id,content,keywords,main_keywords,time
#         X_train_list.append(d.keywords)
#         X_train_time_list.append(d.time)
#
#     corpus = []
#     for i, text in enumerate(X_train_list):
#         text = text.split(' ')
#         corpus.append(TaggedDocument(text, tags=[i]))
#
#     doc2vec_model = Doc2Vec(corpus, min_count=1, window=50, size=100,
#                             sample=1e-5, workers=2)
#
#     doc2vec_model.train(corpus,
#                         total_examples=doc2vec_model.corpus_count,
#                         epochs=5)
#     encoder_data = doc2vec_model.docvecs.doctag_syn0
#
#     return encoder_data

LEN_OF_TRAIN= 50 #20000 #200

data = getData() # docuemnt集合

# document的数量： 25941
trains = data[:LEN_OF_TRAIN]

tests = data[LEN_OF_TRAIN:80]

texts = [[word for word in jieba.lcut(document.keywords)] for document in tests]

frequency = defaultdict(int)
for text in texts:
    for token in text:
        if token != ',':
            frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
          for text in texts]
dictionary = corpora.Dictionary(texts)
dictionary.save('deerwester.dict')
corpusa = [dictionary.doc2bow(text) for text in texts]
#corpus = corpora.MmCorpus('deerwester.mm')
# id2word = dictionary.token2id
# print(id2word)
# py_corpus=read_train_data(tests)
#ldaseq = LdaSeqModel(corpus=corpusa, time_slice=[15000, 10000, 941], num_topics=10, chunksize=1)
ldaseq = LdaSeqModel(corpus=corpusa, time_slice=[10, 10, 10], id2word=None, alphas=0.01, num_topics=4, initialize='gensim', sstats=None,  obs_variance=0.5, chain_variance=0.005, passes=10, random_state=None, lda_inference_max_iter=25, em_min_iter=6, em_max_iter=20, chunksize=100)
"""
    Dynamic Topic Models
    :param time_slice: 最重要,time_slice = [438, 430, 56]代表着，三个月的时间里，第一个有438篇文章，第二个月有430篇文章，第三个月有456篇文章。当然可以以年月日以及任意计量时间都可以
    :param chain_variance:话题演变的快慢是由参数variance影响着,其实LDA中Beta参数的高斯参数，chain_variance的默认是0.05，提高该值可以让演变加速
    :param initialize:两种训练DTM模型的方式，第一种直接用语料，第二种用已经训练好的LDA中的个别统计参数矩阵给入作训练
    :param num_topics:三个时期10个主题的关键词
    :return:
"""
for i in range(0,3):
    print(ldaseq.print_topics(time=i))

