import os
from my_class import *
from operator import itemgetter
from sklearn.utils import shuffle

class document:

    def __init__(self,story_id,event_id,content,keywords,main_keywords,time,title):
        self.story_id=int(story_id)
        self.event_id=int(event_id)
        self.content=content
        self.keywords=keywords
        self.main_keywords=main_keywords
        self.time=time
        self.title = title

    def display(self):
        print(self.story_id)
        print(self.event_id)
        print(self.main_keywords)
        print (self.keywords)
    def get_keywords(self):
        return self.keywords

    def grt_content(self):
        return self.content

    def get_time(self):
        return self.time

def getData():
    print(os.getcwd())
    fr = open('CNESC.txt')
    next(fr)
    doc = []
    print('start load data...')
    fr.seek(0)
    for line in fr.readlines():
        documenttep = line.strip().split('|')
        keywordstep = str(documenttep[6]).strip().split(',')
        #print(keywordstep)
        main_keywordstep = str(documenttep[7]).strip().split(',')
        #print(main_keywordstep)
        # doc.append(document(documenttep[0], documenttep[1], documenttep[3],
        #                     keywordstep, main_keywordstep, documenttep[5]))
        doc.append(document(documenttep[0], documenttep[1], documenttep[3],
                            documenttep[6], documenttep[7], documenttep[5],documenttep[2]))
        fr.seek(0)
    #print(doc[0].display())
    print('document的数量：', len(doc))  # 25941个document
    print('data load over')

    doc = shuffle(doc)

    return doc

def sort_getData():
    fr = open('CNESC.txt')
    next(fr)
    doc = []
    print('start load data...')
    fr.seek(0)
    for line in fr.readlines():
        documenttep = line.strip().split('|')
        # print(documenttep[7])
        # print(str(documenttep[7]).strip())
        keywordstep = str(documenttep[6]).strip().split(',')
        # print(keywordstep)
        main_keywordstep = str(documenttep[7]).strip().split(',')
        # print(main_keywordstep)
        doc.append(
            document(documenttep[0], documenttep[1], documenttep[3], keywordstep, main_keywordstep, documenttep[5],document[2]))
        fr.seek(0)
    # print(doc[0].display())
    print('document的数量：', len(doc))  # 25941个document
    print('data load over')

    result = sorted([doc for i in range(len(doc))],key=itemgetter(5))

    #results = sorted([(sim.similar0(clusters[i], d), i) for i in range(len(clusters))], reverse=True)

    #result = list(sorted(doc,key=lambda x:x[5]))


    print(len(result))




# fr = open('CNESC.txt','r+')
# next(fr)
# doc = []
# print('start load data...')
# fr.seek(0)
# #print(fr.readline())
# for line in fr.readlines():
#     documenttep = line.strip().split('|')
#     #print(len(documenttep))
#     keywordstep = str(documenttep[6]).split(',')
#     #print(keywordstep)
#     main_keywordstep = str(documenttep[7]).strip().split(',')
#     #print(main_keywordstep)
#     doc.append((document(documenttep[0], documenttep[1], documenttep[3], str(keywordstep), main_keywordstep, documenttep[5])))
#     fr.seek(0)
# print('document的数量：',len(doc)) #25941个document
# #print(doc[20].display())

# from keras.preprocessing import sequence
#
# from keras.layers import Input, Embedding, LSTM, Dense, merge, Merge
# from keras.models import Model
# from sklearn.metrics import average_precision_score
# from keras.layers.core import Dense, Dropout, RepeatVector, Activation, Flatten
#
# EMBED_SIZE = 32
# HIDDEN_SIZE= 16
# MAX_LEN= 20
# BATCH_SIZE = 16
# EPOCHS = 5
#
# vocab_size = 5
#
#
# main_input = Input(shape=(MAX_LEN,), dtype='int32', name='doc')
#
# x = Embedding(output_dim=EMBED_SIZE, input_dim=vocab_size, input_length=MAX_LEN)(main_input)
#
# lstm_out = LSTM(HIDDEN_SIZE)(x)
# print(lstm_out)