#coding=utf-8
from __future__ import division

import model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


from my_class import *

NUM_TRAIN=1000
MAX_LEN = 10

vocab_size =20000   #10000


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


    model = train_funtion([X_train_list,X_train_list1,X_train_time_list,X_train_time_list1], [y_train_story, y_train_story1,y_train_event, y_train_event1] ,
                          vocab_size) #vocab_size是输入的维度

    return model

class joint_similarity:
    def __init__(self,trains,train_funtion):
        self.model = getTrains(trains,train_funtion)


    def similar0(self,source,d,ct,dt): #d与event(cluster)的对比 (clusters[i].getCentre(), d.get_keywords(),clusters[i].get_time(),d.get_time()), i)

        #print('cluster的关键词：',str(source))
        X_test_list = [document2(source).get_k()]
        X_test_list1 = [document2(d).get_k()]
        X_test_time_list = [ct]
        X_test_time_list1 = [dt]
        X_test_list = [X_test_list,X_test_list1,X_test_time_list,X_test_time_list1]
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


    def similar1(self, source, target, st, ct): #对比两个event，(storise[i], c.getCentre(), storise[i].get_time(), c.get_time())

        # X_test_list = [document2(source).get_k()]
        # X_test_list1 = [document2(d).get_k()]
        # X_test_list = [X_test_list, X_test_list1]
        #print('看看：单独的两个测试集X:document：', X_test_list)
        X_test_list = [document2(source.keywords).get_k()]
        #print(target)
        X_test_list1 = [document2(target).get_k()]
        X_test_time_list = [st]
        X_test_time_list1 = [ct]
        X_test_list = [X_test_list,X_test_list1,X_test_time_list,X_test_time_list1]
        #print('看看：单独的两个测试集X:document：', X_test_list)

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

            #print('测试集合：',X_test_list[i])
        #print(X_test_list)
        # for i in range(len(X_test_list[:2])):
        #     X_test_list = X_test_list*
        # for i in range(len(X_test_list[2:])):
        #     X_test_list = X_test_list*

        X_pred = self.model.predict(X_test_list)

        #print('story预测值：', X_pred)

        results = [result[0] for result in X_pred[1]]

        return results[0]


# class NNJointRank:
#     def __init__(self, model):
#         self.model = model
#
#     def create_story(self, tests):
#         X_test, y_test = formatK(tests, self.V)
#         X_test = sequence.pad_sequences(X_test, maxlen=nn_multi.MAX_LEN)
#         X_pred = self.model.predict([X_test, X_test])
#         results = sorted([(result[0], i) for i, result in enumerate(X_pred[1])], reverse=True)
#
#         return [tests[i] for score, i in results]
