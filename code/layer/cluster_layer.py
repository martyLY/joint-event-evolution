#coding=utf-8
from __future__ import print_function
from __future__ import division
from functools import reduce
import re
import tarfile
import math
import numpy as np
import os
import shutil
import numpy as np
from itertools import islice

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence

from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
from sklearn.metrics import average_precision_score
from keras.layers.core import Dense, Dropout, RepeatVector, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

from model import *
from my_class import *
from evau import *

# EMBED_SIZE = 32
# HIDDEN_SIZE= 16
# MAX_LEN= 200
# BATCH_SIZE = 16
# EPOCHS = 5

EMBED_SIZE = 32
HIDDEN_SIZE= 16
MAX_LEN= 200
BATCH_SIZE = 2 #16
EPOCHS = 4

vocab_size = 2000 # 单词数量

def getMostSimilarClusters(clusters,d,sim):
    print('clusters',clusters)
    print('d',d)
    results = sorted([(sim.similar0(clusters[i].getCentre(), d.get_keywords()), i) for i in range(len(clusters))], reverse=True)
    # 排序从大到小，对比document和cluster(event)的相似，20个event的相似度 (相似度，i)
    similar, cIndex = results[0]  # 当前相似度
    currentSimilarList = [results[i][0] for i in range(len(results))]  # 当前相似度排名

def addToOldCluster(clusters,cIndex,d): #找到它要加入的cluster
    c=clusters[cIndex]
    c.add(d)
    clusters.remove(c) #把c的列表元素删掉
    clusters.insert(0,c)#把c放在列表的第一个

def addToNewCluster(clusters,cIndex,d): #新建一个cluster，然后把document加入这个document，然后把该document的keywords更新一下
    c=cluster()
    c.add(d)
    #del clusters[-1]
    clusters.insert(0,c)# 把新建的这个cluster加入clusters大家庭

def clustersDetection(X_test_list,y_test_list,k,sim): # (X_test_list,y_test_list, 10, NE)
    clusters = [cluster() for i in range(k)]
    similarList = SimilarList()
    avg = 0
    sd = 0

    X_pre = sim.model.predict(X_test_list)
    #print('???1111111111',X_pre[0])
    results = [result[0] for result in X_pre]
    #print('输出列表',results)
    print('样本属于每一个类别的概率,预测值：', X_pre)

    # X_fen = np.argmax(X_pre,axis=1)
    # print(X_fen)

    score, acc = sim.model.evaluate(X_test_list, y_test_list, batch_size=BATCH_SIZE)
    print('*******************',score,acc)

    similar, cIndex = results[0]  # 当前相似度
    currentSimilarList = [results[i][0] for i in range(len(results))]  # 当前相似度排名

    similarList.addRange(currentSimilarList)
    print('是否为同一event的阀值：')
    print(similar, avg - 3 * sd)
    # if similar<avg:
    #     addToNewCluster(clusters,cIndex,d)
    # else:
    #     addToOldCluster(clusters,cIndex,d)
    # avg,sd=similarList.getAvgSD()
    # print('更新之后的：',avg,sd)

    for c in clusters:
        print(len(c.documents))
    #eval_clusters(clusters)

    return clusters

def clustersDetection(tests,k,sim):
    clusters = [cluster() for i in range(k)]
    similarList = SimilarList()
    avg = 0
    sd = 0
    eva_list = []

    print(len(tests))

    for i,d in enumerate(tests):
        results = sorted([(sim.similar0(clusters[i].getCentre(), d.get_keywords(),clusters[i].get_time(),d.get_time()), i) for i in range(len(clusters))],
                         reverse=True) #document与k个clusters的相似度排名，从高到低
        similar, cIndex = results[0]
        currentSimilarList = [results[i][0] for i in range(len(results))]
        #print('00000000',currentSimilarList)
        similarList.addRange(currentSimilarList)
        if similar<avg:
            addToNewCluster(clusters,cIndex,d)
        else:
            addToOldCluster(clusters,cIndex,d)
        avg,sd=similarList.getAvgSD()
        eva_list.append(avg - 3 * sd)

    sorted(eva_list)

    print('event_id正确率：', eva_list)


    # for c in clusters:
    #     print('每个集合(event)的个数：',len(c.documents))
    #     c.sort_cluster()
        #print('该event的关键词：')
        #c.showCentre()


    #eval_cmin(clusters)
    eval_cluster(clusters)




    return clusters
