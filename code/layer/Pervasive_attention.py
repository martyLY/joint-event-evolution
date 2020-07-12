from keras.layers import Input, Embedding, \
    Lambda, Concatenate, BatchNormalization, \
    Conv2D, Dropout, Dense, MaxPool2D, ZeroPadding2D, \
    AveragePooling2D, ZeroPadding1D
from keras.layers import Activation, TimeDistributed, Conv1D
from keras.models import Model
import keras.backend as K
from keras import optimizers

max_enc_len = 200
max_dec_len = 200

src_word_num = 1500
embedding_dim = 32
tgt_word_num = 1500

conv_emb_layers = 2
# Inputs
src_input = Input(shape=(max_enc_len,), name='src_input')
tgt_input = Input(shape=(max_dec_len,), name='tgt_input')

# embedding
src_embedding = Embedding(src_word_num + 2,
                          embedding_dim,
                          name='src_embedding')(src_input)
tgt_embedding = Embedding(tgt_word_num + 2,
                          embedding_dim,
                          name='tgt_embedding')(tgt_input)

# implement a convEmbedding
for i in range(conv_emb_layers):
    src_embedding = Conv1D(embedding_dim, 3, padding='same',
                           data_format='channels_last', activation='relu')(src_embedding)
    tgt_embedding = ZeroPadding1D(padding=(2, 0))(tgt_embedding)
    tgt_embedding = Conv1D(embedding_dim, 3, padding='valid',
                           data_format='channels_last', activation='relu')(tgt_embedding)


def src_reshape_func(src_embedding, repeat):
    """
    对embedding之后的source sentence的tensor转换成pervasive-attention model需要的shape
    arxiv.org/pdf/1808.03867.pdf
    :param src_embedding: source sentence embedding之后的结果[tensor]
    :param repeat: 需要重复的次数, target sentence t的长度[int]
    :return: 2D tensor (?, s, t, embedding_dim)
    """
    input_shape = src_embedding.shape
    src_embedding = K.reshape(src_embedding, [-1, 1, input_shape[-1]])
    src_embedding = K.tile(src_embedding, [1, repeat, 1])
    src_embedding = K.reshape(src_embedding, [-1, input_shape[1], repeat, input_shape[-1]])

    return src_embedding


def tgt_reshape_func(tgt_embedding, repeat):
    """
    对embedding之后的target sentence的tensor转换成pervasive-attention model需要的shape
    arxiv.org/pdf/1808.03867.pdf
    :param tgt_embedding: target sentence embedding之后的结果[tensor]
    :param repeat: 需要重复的次数, source sentence s的长度[int]
    :return: 2D tensor (?, s, t, embedding_dim)
    """
    input_shape = tgt_embedding.shape
    tgt_embedding = K.reshape(tgt_embedding, [-1, 1, input_shape[-1]])
    tgt_embedding = K.tile(tgt_embedding, [1, repeat, 1])
    tgt_embedding = K.reshape(tgt_embedding, [-1, input_shape[1], repeat, input_shape[-1]])
    tgt_embedding = K.permute_dimensions(tgt_embedding, [0, 2, 1, 3])

    return tgt_embedding


def src_embedding_layer(src_embedding, repeat):
    """
    转换成Lambda层
    :param src_embedding: source sentence embedding之后的结果[tensor]
    :param repeat: 需要重复的次数, target sentence t的长度[int]
    :return: 2D tensor (?, s, t, embedding_dim)
    """
    return Lambda(src_reshape_func,
                  arguments={'repeat': repeat})(src_embedding)


def tgt_embedding_layer(tgt_embedding, repeat):
    """
    转换层Lambda层
    :param tgt_embedding: target sentence embedding之后的结果[tensor]
     :param repeat: 需要重复的次数, target sentence t的长度[int]
    :return: 2D tensor (?, s, t, embedding_dim)
    """
    return Lambda(tgt_reshape_func,
                  arguments={'repeat': repeat})(tgt_embedding)


# concatenate
src_embedding = src_embedding_layer(src_embedding, repeat=max_dec_len)
tgt_embedding = tgt_embedding_layer(tgt_embedding, repeat=max_enc_len)
src_tgt_embedding = Concatenate(axis=3)([src_embedding, tgt_embedding])


# transition layer
def transition_block(x,
                     reduction):
    """A transition block.
    该transition block与densenet的标准操作不一样，此处不包括pooling层
    pervasive-attention model中的transition layer需要保持输入tensor
    的shape不变 arxiv.org/pdf/1808.03867.pdf
    # Arguments
        x: input tensor.
        reduction: float, the rate of feature maps need to retain.
    # Returns
        output tensor for the block.
    """
    x = BatchNormalization(axis=3, epsilon=1.001e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(int(K.int_shape(x)[3] * reduction), 1, use_bias=False)(x)

    x = MaxPool2D((2, 1), strides=(2, 1))(x)

    return x


# building block
def conv_block(x,
               growth_rate,
               dropout):
    """A building block for a dense block.
    该conv block与densenet的标准操作不一样，此处通过
    增加Zeropadding2D层实现论文中的mask操作，并将
    Conv2D的kernel size设置为(3, 2)
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        dropout: float, dropout rate at dense layers.
    # Returns
        Output tensor for the block.
    """
    x1 = BatchNormalization(axis=3,
                            epsilon=1.001e-5)(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False)(x1)
    x1 = BatchNormalization(axis=3, epsilon=1.001e-5)(x1)
    x1 = Activation('relu')(x1)
    x1 = ZeroPadding2D(padding=((1, 1), (1, 0)))(x1)  # mask sake
    x1 = Conv2D(growth_rate, (3, 2), padding='valid')(x1)
    x1 = Dropout(rate=dropout)(x1)

    x = Concatenate(axis=3)([x, x1])

    return x


# dense block
def dense_block(x,
                blocks,
                growth_rate,
                dropout):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        growth_rate:float, growth rate at dense layers.
        dropout: float, dropout rate at dense layers.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, growth_rate=growth_rate, dropout=dropout)

    return x


# # densenet 4 dense block
# if len(blocks) == 1:
#     x = dense_block(x, blocks=blocks[-1], growth_rate=growth_rate, dropout=dropout)
# else:
#     for i in range(len(blocks) - 1):
#         x = dense_block(x, blocks=blocks[i], growth_rate=growth_rate, dropout=dropout)
#         x = transition_block(x, reduction)
#     x = dense_block(x, blocks=blocks[-1], growth_rate=growth_rate, dropout=dropout)

# pervasive-attention model

def h_max_pooling_layer(h):
    """
    实现论文中提到的最大池化 arxiv.org/pdf/1808.03867.pdf
    :param h: 由densenet结构输出的shape为(?, s, t, fl)的tensor[tensor]
    :return: (?, t, fl)
    """
    h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(h)
    h = MaxPool2D(data_format='channels_first',
                  pool_size=(h.shape[2], 1))(h)
    h = Lambda(lambda x: K.squeeze(x, axis=2))(h)

    return h


def pervasive_attention(blocks,
                        conv1_filters=64,
                        growth_rate=12,
                        reduction=0.5,
                        dropout=0.2,
                        max_enc_len=200,
                        max_dec_len=200,
                        embedding_dim=128,
                        src_word_num=4000,
                        tgt_word_num=4000,
                        samples=12000,
                        batch_size=8,
                        conv_emb_layers=6
                        ):
    """
    build a pervasive-attention model with a densenet-like cnn structure.
    :param blocks: a list with length 4, indicates different number of
        building blocks in 4 dense blocks, e.g which [6, 12, 48, 32]
        for DenseNet201 and [6, 12, 32, 32] for DenseNet169. [list]
    :param conv1_filters: the filters used in first 1x1 conv to
        reduce the channel size of embedding input. [int]
    :param growth_rate: float, growth rate at dense layers. [int]
    :param reduction: float, the rate of feature maps which
        need to retain after transition layer. [float]
    :param dropout: dropout rate used in each conv block, default 0.2. [float]
    :param max_enc_len: the max len of source sentences. [int]
    :param max_dec_len: the max len of target sentences. [int]
    :param embedding_dim: the hidden units of first two embedding layers. [int]
    :param src_word_num: the vocabulary size of source sentences. [int]
    :param tgt_word_num: the vocabulary size of target sentences. [int]
    :param samples: the size of the training data. [int]
    :param batch_size: batch size. [int]
    :param conv_emb_layers: the layers of the convolution embedding. [int]
    :return:
    """
    # Inputs
    src_input = Input(shape=(max_enc_len,), name='src_input')
    tgt_input = Input(shape=(max_dec_len,), name='tgt_input')

    # embedding
    src_embedding = Embedding(src_word_num + 2,
                              embedding_dim,
                              name='src_embedding')(src_input)
    tgt_embedding = Embedding(tgt_word_num + 2,
                              embedding_dim,
                              name='tgt_embedding')(tgt_input)
    # implement a convEmbedding
    for i in range(conv_emb_layers):
        src_embedding = Conv1D(embedding_dim, 3, padding='same',
                               data_format='channels_last', activation='relu')(src_embedding)
        tgt_embedding = ZeroPadding1D(padding=(2, 0))(tgt_embedding)
        tgt_embedding = Conv1D(embedding_dim, 3, padding='valid',
                               data_format='channels_last', activation='relu')(tgt_embedding)

    # concatenate
    src_embedding = src_embedding_layer(src_embedding, repeat=max_dec_len)
    tgt_embedding = tgt_embedding_layer(tgt_embedding, repeat=max_enc_len)
    src_tgt_embedding = Concatenate(axis=3)([src_embedding, tgt_embedding])

    # densenet conv1 1x1
    x = Conv2D(conv1_filters, 1, strides=1)(src_tgt_embedding)
    x = BatchNormalization(axis=3, epsilon=1.001e-5)(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 1), strides=(2, 1))(x)

    # densenet 4 dense block
    if len(blocks) == 1:
        x = dense_block(x, blocks=blocks[-1], growth_rate=growth_rate, dropout=dropout)
    else:
        for i in range(len(blocks) - 1):
            x = dense_block(x, blocks=blocks[i], growth_rate=growth_rate, dropout=dropout)
            x = transition_block(x, reduction)
        x = dense_block(x, blocks=blocks[-1], growth_rate=growth_rate, dropout=dropout)

    # Max pooling
    h = h_max_pooling_layer(x)

    # Target sequence prediction
    output = Dense(tgt_word_num + 2, activation='softmax')(h)

    # compile
    model = Model([src_input, tgt_input], [output])
    # adam = optimizers.Adam(lr=0.0001,
    #                        beta_1=0.9,
    #                        beta_2=0.999,
    #                        epsilon=1e-08,
    #                        decay=0.05 * batch_size / samples)
    # model.compile(optimizer=adam, loss='categorical_crossentropy')

    return model



