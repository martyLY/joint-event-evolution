#coding=utf-8
from __future__ import print_function
from __future__ import division
from functools import reduce
import re
import tarfile
import math

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence

from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
from sklearn.metrics import average_precision_score
from keras.layers.core import Dense, Dropout, RepeatVector, Activation, Flatten
from keras.layers import concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from data_loader import *
from cluster_layer import *
from fir_layer import *
import similar
from story_layer import *

from sklearn.decomposition import PCA


EMBED_SIZE = 32 #32、3
HIDDEN_SIZE= 16 #16、3
MAX_LEN= 200 #200 、 10
BATCH_SIZE = 32 #16、1
EPOCHS = 4

EVENT_SIZE = 4000
STORY_SIZE = 3000

vocab_size = 20000 # 单词数量


def joint_train(X_train_list, y_train_list ,vocab_size): #X_train_list, [y_train_story, y_train_event] ,vocab_size

    N = len(X_train_list)
    # N1 = len(X_train_title_list)
    # print('标题训练集的长，个数：',N1)

    sum = 0

    for i in range(len(X_train_list)):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train_list[i])
        # tokenizer.word_index

        sum += len(tokenizer.word_index)

        X_train_list[i] = tokenizer.texts_to_sequences(X_train_list[i])
        X_train_list[i] = sequence.pad_sequences(X_train_list[i], maxlen=200)



    y_train_list = [np.array(y_train) for y_train in y_train_list]

    #print(y_train_list)

    input_list = []
    out_list = []
    for i in range(N):
        input, out = get_lstm_input_output('f%d' % i, vocab_size)
        input_list.append(input)
        out_list.append(out)

    # Event loss
    event_loss0 = Dense(1, activation='sigmoid', name='event_output_0')(out_list[0])
    event_loss1 = Dense(1, activation='sigmoid', name='event_output_1')(out_list[1])

    summ_out0 = concatenate([out_list[0],event_loss0])
    summ_out1 = concatenate([out_list[1],event_loss1])

    # summ_out0 = merge([out_list[0], event_loss0], mode='concat')
    # summ_out1 = merge([out_list[1], event_loss1], mode='concat')
    summary_loss0 = Dense(1, activation='sigmoid', name='summary_output_0')(summ_out0)
    summary_loss1 = Dense(1, activation='sigmoid', name='summary_output_1')(summ_out1)

    model= Model(inputs=input_list, outputs=[summary_loss0,summary_loss1,event_loss0,event_loss1] , name='joint_train')

    print(model.summary())

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    print('训练集X：',X_train_list)
    print('训练集y:',y_train_list)

    # batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
    # nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次。
    model.fit(X_train_list, y_train_list, epochs=EPOCHS, batch_size=BATCH_SIZE)


    return model


def pure_LSTM_train(X_train_list, y_train_list, vocab_size):
    N = len(X_train_list)

    for i in range(len(X_train_list)):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train_list[i])
        # tokenizer.word_index


        X_train_list[i] = tokenizer.texts_to_sequences(X_train_list[i])
        X_train_list[i] = sequence.pad_sequences(X_train_list[i], maxlen=200)

    y_train_list = [np.array(y_train) for y_train in y_train_list]

    # print(y_train_list)

    input_list = []
    out_list = []
    for i in range(N):
        input, out = get_lstm_input_output('f%d' % i, vocab_size)
        input_list.append(input)
        out_list.append(out)

    x = concatenate(out_list)

    # flatten = Reshape((180,)) (merged)

    # x= RepeatVector(HIDDEN_SIZE)(x)

    # x = LSTM(HIDDEN_SIZE)(x)

    main_loss = Dense(1, activation='sigmoid', name='main_output')(x)

    model = Model(input=input_list, output=main_loss)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    model.fit(X_train_list, y_train_list, nb_epoch=EPOCHS, batch_size=BATCH_SIZE)

    return model


def simple_joint_train(X_train_list, y_train_list, vocab_size):
    N = len(X_train_list)

    for i in range(len(X_train_list)):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train_list[i])
        # tokenizer.word_index
        # del tokenizer.word_index['，']
        # del tokenizer.word_index['。']
        # del tokenizer.word_index['的']
        # del tokenizer.word_index['、']
        # del tokenizer.word_index['【']
        # del tokenizer.word_index['】']
        # del tokenizer.word_index['/']

        X_train_list[i] = tokenizer.texts_to_sequences(X_train_list[i])
        X_train_list[i] = sequence.pad_sequences(X_train_list[i], maxlen=200)

    y_train_list = [np.array(y_train) for y_train in y_train_list]

    # print(y_train_list)

    input_list = []
    out_list = []
    for i in range(N):
        input, out = get_lstm_input_output('f%d' % i, vocab_size)
        input_list.append(input)
        out_list.append(out)

    x = concatenate(out_list)

    summary_loss0 = Dense(1, activation='sigmoid', name='summary_output_0')(out_list[0])
    summary_loss1 = Dense(1, activation='sigmoid', name='summary_output_1')(out_list[1])
    event_loss0 = Dense(1, activation='sigmoid', name='event_output_0')(out_list[0])
    event_loss1 = Dense(1, activation='sigmoid', name='event_output_1')(out_list[1])

    model = Model(input=input_list, output=[summary_loss0, summary_loss1, event_loss0, event_loss1])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    model.fit(X_train_list, y_train_list, nb_epoch=EPOCHS, batch_size=BATCH_SIZE)

    return model


def event_similar_train(X_train_list, y_train_list, vocab_size): #只有event层
    N = len(X_train_list)

    for i in range(len(X_train_list)):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train_list[i])
        # tokenizer.word_index
        # del tokenizer.word_index['，']
        # del tokenizer.word_index['。']
        # del tokenizer.word_index['的']
        # del tokenizer.word_index['、']
        # del tokenizer.word_index['【']
        # del tokenizer.word_index['】']
        # del tokenizer.word_index['/']

        X_train_list[i] = tokenizer.texts_to_sequences(X_train_list[i])
        X_train_list[i] = sequence.pad_sequences(X_train_list[i], maxlen=200)

    y_train_list = [np.array(y_train) for y_train in y_train_list]

    # print(y_train_list)

    input_list = []
    out_list = []
    for i in range(N):
        input, out = get_lstm_input_output('f%d' % i, vocab_size)
        input_list.append(input)
        out_list.append(out)


    event_loss0 = Dense(1, activation='sigmoid', name='event_output_0')(out_list[0])
    event_loss1 = Dense(1, activation='sigmoid', name='event_output_1')(out_list[1])


    model = Model(input=input_list, output=[event_loss0, event_loss1])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    model.fit(X_train_list, y_train_list, nb_epoch=EPOCHS, batch_size=BATCH_SIZE)

    return model


def story_similar_train(X_train_list, y_train_list, vocab_size): #只有story层
    N = len(X_train_list)

    for i in range(len(X_train_list)):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train_list[i])
        # tokenizer.word_index
        # del tokenizer.word_index['，']
        # del tokenizer.word_index['。']
        # del tokenizer.word_index['的']
        # del tokenizer.word_index['、']
        # del tokenizer.word_index['【']
        # del tokenizer.word_index['】']
        # del tokenizer.word_index['/']

        X_train_list[i] = tokenizer.texts_to_sequences(X_train_list[i])
        X_train_list[i] = sequence.pad_sequences(X_train_list[i], maxlen=200)

    y_train_list = [np.array(y_train) for y_train in y_train_list]

    # print(y_train_list)

    input_list = []
    out_list = []
    for i in range(N):
        input, out = get_lstm_input_output('f%d' % i, vocab_size)
        input_list.append(input)
        out_list.append(out)

    summ_out0 = concatenate(out_list[0])
    summ_out1 = concatenate(out_list[1])
    summary_loss0 = Dense(1, activation='sigmoid', name='summary_output_0')(summ_out0)
    summary_loss1 = Dense(1, activation='sigmoid', name='summary_output_1')(summ_out1)

    model = Model(input=input_list, output=[summary_loss0, summary_loss1])

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    model.fit(X_train_list, y_train_list, nb_epoch=EPOCHS, batch_size=BATCH_SIZE)

    return model

def get_test(tests):
    X_test_list = []
    X_test_list1 = []
    y_test_story = []
    y_test_story1 = []
    y_test_event = []
    y_test_event1 = []
    i1 = 0
    for d in tests:
        if i1==0:

            X_test_list.append(d.content)
            y_test_story.append(d.story_id)
            y_test_event.append(d.event_id)
            i1 = 1
        else:

            X_test_list1.append(d.content)
            y_test_story1.append(d.story_id)
            y_test_event1.append(d.event_id)
            i1 = 0

    X_test_list,y_test_list= [X_test_list, X_test_list1], [y_test_story, y_test_story1, y_test_event, y_test_event1]

    #print('test_yyyyyyyyyyy',y_test_list)

    for i in range(len(X_test_list)):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_test_list[i])
        # tokenizer.word_index


        X_test_list[i] = tokenizer.texts_to_sequences(X_test_list[i])
        X_test_list[i] = sequence.pad_sequences(X_test_list[i], maxlen=10)

    y_test_list = [np.array(y_test) for y_test in y_test_list]

    print('测试集的X：', X_test_list)
    print('测试集合的y：',y_test_list)


    return X_test_list,y_test_list

def JEDS(trains,tests):

    # 选择训练模型：

    # joint_train:我们提出的模型，LSTM输入，AM机制处理，第一层event集合，第二层story集合
    # pure_LSTM_train：LSTM-Pipeline. In this setting, there is no parameter sharing, and a separate LSTM representation is used in each subtask
    # simple_joint_train :在训练中简单地集合所有的parameters
    # event_similar_train :+event
    # story_similar_train :+story

    NE = similar.joint_similarity(trains,joint_train)


    clusters = clustersDetection(tests, EVENT_SIZE, NE)


    storise = storiseDetection(clusters,tests, STORY_SIZE, NE)
