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

class tdocument:

    def __init__(self,story_id,event_id,time,title):
        self.story_id=int(story_id)
        self.event_id=int(event_id)
        self.time=time
        self.title = title

    def display(self):
        print(self.story_id)
        print(self.event_id)

    def get_time(self):
        return self.time

def getData():
    print(os.getcwd())
    #fr = open('CNESC.txt')
    fr = open('000_lable.txt')
    next(fr)
    doc = []
    print('start load data...')
    fr.seek(0)
    for line in fr.readlines():
        documenttep = line.strip().split('|')
        if(len(documenttep)==8):
            keywordstep = str(documenttep[6]).strip().split(',')
            # #print(keywordstep)
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

def getExprimentData():
    fr = open('0.txt')
    next(fr)
    doc = []
    print('start load expriment data...')
    fr.seek(0)
    for line in fr.readlines():
        documenttep = line.strip().split('|')
        doc.append(tdocument(documenttep[0], documenttep[1], documenttep[2], documenttep[3]))
        fr.seek(0)
    print('expriment docuemnt的数量：',len(doc))
    print('expriment data over')

    #doc = shuffle(doc)

    return doc
