import math
import string
import model

class document:

    def __init__(self,story_id,event_id,content,keywords,main_keywords,time,):
        self.story_id=story_id
        self.event_id=event_id
        self.content=content
        self.keywords=keywords
        self.main_keywords=main_keywords
        self.time=time
        #self.title=title

    def get_keywords(self):
        return self.keywords

    def get_cotent(self):
        return self.content

class document2:
    def __init__(self,keywords):
        self.keywords = keywords
        #self.content = keywords
    def get_k(self):
        a = []
        a.append(self.keywords)
        return self.keywords

class cluster:
    def __init__(self):
        #self.story_id = story_id
        self.documents = []
        self.keywords = ''
        self.time = ['0']

    def getCentre(self):
        return self.keywords

    def updateTime(self):
        self.documents = sorted([self.documents for document.time in self.documents])

    def updateCentre(self,d):
        # centre = []
        # for document in self.documents:
        #     for word in document.main_keywords:
        #         if word not in centre:
        #             centre[word]=0
        #         centre[word] += 1
        self.keywords.append(d.get_keywords)
        for k in d.get_keywords:
            k.translate(str.maketrans('', '', string.punctuation))

    def add(self,d):
        self.documents.append(d)
        self.time.append(d.time)

        self.keywords += d.get_keywords()
        #self.updateCentre(d)
        #self.updateTime()
    def showCentre(self):
        print(self.keywords)
    def get_keywords(self):
        return self.keywords
    def get_time(self):
        return self.time[0]

class story:
    def __init__(self):
        #self.story_id = story_id
        self.documents = []
        self.keywords = ''
        self.cluters = []
        self.time = ['0']

    def add(self,c):
        self.cluters.append(c)
        self.keywords += c.get_keywords()
        self.time.append(c.get_time())


    def showCentre(self):
        print(self.keywords)

    def showDocument(self):
        for c in self.cluters:
            for d in c.documents:
                print(d.content)

    def get_time(self):
        return self.time[0]



class SimilarList:
    def __init__(self):
        self.a = []

    def getCentre(self):
        return self.keywords

    def getAvgSD(self):  # avg-sd
        n = len(self.a)
        if n > 0:
            avg = sum(self.a) / n
            sd = math.sqrt(sum([(i - avg) ** 2 for i in self.a]) / n)
            return avg, sd
        else:
            return 0, 0

    def addRange(self,b):
        self.a+=[i for i in b if i>0]


class Cosine:
    def __init__(self):
        pass

    def similar(self, source, target):
        numerator = sum([source[word] * target[word] for word in source if word in target])
        sourceLen = math.sqrt(sum([value * value for value in source.values()]))
        targetLen = math.sqrt(sum([value * value for value in target.values()]))
        denominator = sourceLen * targetLen
        if denominator == 0:
            return 0
        else:
            return numerator / denominator
