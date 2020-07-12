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
import scipy.stats
from operator import itemgetter
from itertools import islice

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence

from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
from sklearn.metrics import average_precision_score
from keras.layers.core import Dense, Dropout, RepeatVector, Activation, Flatten


from model import *
from my_class import *
from similar import *
from output_txt import *
import scipy.stats

def getMostSimilarStorise(storise, c,sim):
    results = sorted([(sim.similar1(storise[i], c), i) for i in range(len(storise))], reverse=True)
    similar, cIndex = results[0]  # 当前相似度
    currentSimilarList = [results[i][0] for i in range(len(results))]  # 当前相似度排名


def addToOldStory(storise,sIndex,c):
    c1=storise[sIndex]
    c1.add(c)
    storise.remove(c1)
    storise.insert(0,c1)

def addToNewStory(storise,sIndex,c):
    c1=story()
    c1.add(c)
    #del storise[-1]
    storise.insert(0,c1)


def storiseDetection(clusters,tests,k, sim):
    clusters = [c for c in clusters if len(c.documents) > 0]
    #print('event的个数：',len(clusters))
    storise = [story() for i in range(k)]
    similarList = SimilarList()
    avg = 0
    cor_sum = 0
    eva_list = []
    for i,c in enumerate(clusters):
        #similar, sIndex, currentSimilarList = getMostSimilarStorise(storise, c.getCentre(), sim)
        results = sorted([(sim.similar1(storise[i], c.getCentre(), storise[i].get_time(), c.get_time()), i) for i in range(len(storise))], reverse=True)
        similar, sIndex = results[0]  # 当前相似度
        currentSimilarList = [results[i][0] for i in range(len(results))]  # 当前相似度排名
        similarList.addRange(currentSimilarList)

        if similar < avg:
            addToNewStory(storise, sIndex, c )
        else:
            addToOldStory(storise, sIndex, c)
        avg, sd = similarList.getAvgSD()
        eva_list.append(avg + 10 * sd)

    readResult(eva_list)
    sorted(eva_list)
    for cor in eva_list[:2]:
        cor_sum +=cor
    re = cor_sum/len(eva_list[:])

    print('story_id正确率',re)


    eval_story(storise)
    print('story的个数：',len(storise))



    out_txt_story(storise)
    out_sort_story()

    # for s in storise:
    #     for c in s.cluters:
    #         for d in c.documents:
    #             #d.dsort(c.documents)
    #             print(str(d.story_id) + '|' + str(d.event_id) + '|' + str(d.title) + '|' + str(d.get_keywords()) + '|' + str(d.time) )

    return storise

