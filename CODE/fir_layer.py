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
from att_layer import *
from keras.preprocessing import sequence

from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
from sklearn.metrics import average_precision_score
from keras.layers.core import Dense, Dropout, RepeatVector, Activation, Flatten, Reshape
from keras import backend as K
from keras.layers import concatenate
from keras.preprocessing.text import Tokenizer
from keras.layers.wrappers import Bidirectional

from model import *
from my_class import *
from sklearn.decomposition import PCA

EMBED_SIZE = 32 #32 ，嵌入后的维度
HIDDEN_SIZE= 16 # 隐藏层数量
MAX_LEN= 200 #200,5，输入是几维度
BATCH_SIZE = 16
EPOCHS = 5


vocab_size = 2001 # 单词数量

from keras import initializers
import keras.backend as K
from keras.layers import Permute,Multiply

def attention_3d_block(inputs, single_attention_vector=False):
    # 如果上一层是LSTM，需要return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # 乘上了attention权重，但是并没有求和，好像影响不大
    # 如果分类任务，进行Flatten展开就可以了
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def get_lstm_input_output(part_name, vocab_size):

    main_input = Input(shape=(MAX_LEN,), dtype='float32', name=part_name + '_input') #

    x = Embedding(output_dim=EMBED_SIZE, input_dim=vocab_size, input_length=MAX_LEN)(main_input)

    lstm_out = LSTM(HIDDEN_SIZE)(x)

    #lstm_out = Bidirectional(GRU(32, return_sequences=True))(x)

    #lstm_out = Bidirectional(LSTM(32))(x)

    #l_att = attention_3d_block(lstm_out)

    #l_out = GRU(1)(l_att)

    #l_att = AttentionLayer()(lstm_out)


    return main_input, lstm_out
