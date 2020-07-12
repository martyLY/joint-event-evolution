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

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence

from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model
from sklearn.metrics import average_precision_score
from keras.layers.core import Dense, Dropout, RepeatVector, Activation, Flatten, Reshape
from keras import backend as K
from keras.layers import concatenate
from keras.preprocessing.text import Tokenizer

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

# Attention GRU network
class AttentionLayer():
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

def get_lstm_input_output(part_name, vocab_size):

    main_input = Input(shape=(MAX_LEN,), dtype='float32', name=part_name + '_content_input') #

    x = Embedding(output_dim=EMBED_SIZE, input_dim=vocab_size, input_length=MAX_LEN)(main_input)

    lstm_out = LSTM(HIDDEN_SIZE)(x)

    #l_att = AttentionLayer()(lstm_out)


    return main_input, lstm_out
