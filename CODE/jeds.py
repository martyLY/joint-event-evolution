import random
from cluster_layer import *
from model import *

EMBED_SIZE = 32
HIDDEN_SIZE= 16
MAX_LEN= 100
BATCH_SIZE = 2 #16
EPOCHS = 4

vocab_size = 2000 # 单
class jeds_similarity:

    def __init__(self,trains,train_funtion):
        self.model = getTrains(trains,train_funtion)

    def similar0(self,source,d,ct,dt): #d与event(cluster)的对比 (clusters[i].getCentre(), d.get_keywords(),clusters[i].get_time(),d.get_time()), i)

        #print('cluster的关键词：',str(source))
        X_test_list = [document2(source).get_k()]
        X_test_list1 = [document2(d).get_k()]
        X_test_time_list = [ct]
        X_test_time_list1 = [dt]
        X_test_list = [X_test_list,X_test_list1]
        #print('看看：单独的两个测试集X:document：',X_test_list)

        for i in range(len(X_test_list)):
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(X_test_list[i])
            # tokenizer.word_index
            # del tokenizer.word_index['，']
            # del tokenizer.word_index['。']
            # del tokenizer.word_index['的']
            # del tokenizer.word_index['、']
            # del tokenizer.word_index['【']
            # del tokenizer.word_index['】']
            # del tokenizer.word_index['/']

            X_test_list[i] = tokenizer.texts_to_sequences(X_test_list[i])
            X_test_list[i] = sequence.pad_sequences(X_test_list[i], maxlen=200)

        #print(X_test_list)

        X_pred = self.model.predict(X_test_list)

        results = [result[0] for result in X_pred[0]]
        return results[0]

def jeds_train(X_train_list, y_train_list ,vocab_size): #X_train_list, [y_train_story, y_train_event] ,vocab_size

    N = len(X_train_list)
    sum = 0

    for i in range(len(X_train_list)):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train_list[i])
        # tokenizer.word_index
        sum += len(tokenizer.word_index)
        X_train_list[i] = tokenizer.texts_to_sequences(X_train_list[i])
        X_train_list[i] = sequence.pad_sequences(X_train_list[i], maxlen=200)

    print('tokens的个数：',sum)

    y_train_list = [np.array(y_train) for y_train in y_train_list]

    input_list = []
    out_list = []
    for i in range(N):
        input, out = get_lstm_input_output('f%d' % i, vocab_size)
        input_list.append(input)
        out_list.append(out)

    # Event loss
    event_loss0 = Dense(1, activation='sigmoid', name='event_output_0')(out_list[0])
    event_loss1 = Dense(1, activation='sigmoid', name='event_output_1')(out_list[1])

    summary_loss0 = Dense(1, activation='sigmoid', name='summary_output_0')(event_loss0)
    summary_loss1 = Dense(1, activation='sigmoid', name='summary_output_1')(event_loss1)

    model= Model(inputs=input_list, outputs=[summary_loss0,summary_loss1,event_loss0,event_loss1] , name='joint_train')

    print(model.summary())

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    #model.compile(optimizer=Nadam(lr=0.0001), loss='binary_crossentropy' ,metrics=['accuracy'])

    # print('训练集X：',X_train_list)
    # print('训练集y:',y_train_list)

    # batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
    # nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次。
    model.fit(X_train_list, y_train_list, validation_split=0.9, epochs=EPOCHS, batch_size=BATCH_SIZE)

    #model.save_weights('joint_train_weight.h5')

    return model

def getTrains(trains,train_funtion):
    X_train_list = []
    X_train_list1 = []
    X_train_time_list = []
    X_train_time_list1 = []
    X_train_title_list = []
    y_train_story = []
    y_train_story1= []
    y_train_event = []
    y_train_event1 = []

    i = 0

    for d in trains:
        #story_id,event_id,content,keywords,main_keywords,time
        if i==0:
            X_train_list.append(d.keywords)
            X_train_time_list.append(d.time)
            y_train_story.append(d.story_id)
            y_train_event.append(d.event_id)
            i = 1
        else:
            X_train_list1.append(d.keywords)
            X_train_time_list1.append(d.time)
            y_train_story1.append(d.story_id)
            y_train_event1.append(d.event_id)
            i = 0


    model = train_funtion([X_train_list,X_train_list1], [y_train_story, y_train_story1,y_train_event, y_train_event1] ,
                          vocab_size) #vocab_size是输入的维度

    return model

def JEDS(trains, tests):
    ISim = jeds_similarity(trains,jeds_train)
    #clusters = clustersDetection(tests, EVENT_SIZE, NE)
    clusters = clustersDetection(tests, EVENT_SIZE, ISim)
