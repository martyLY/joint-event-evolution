# coding=utf-8
from __future__ import division
import math
from my_class import *
from evau import *
import random

# Petrovic et al., (NAACL, 2010). Streaming first story detection with application to Twitter

t = 0.5

class Cosine:
    def __init__(self):
        pass

    def similar(self, source, target):
        numerator = sum([source[keywords] * target[keywords] for keywords in source if keywords in target])
        sourceLen = math.sqrt(sum([value * value for value in source.values()]))
        targetLen = math.sqrt(sum([value * value for value in target.values()]))
        denominator = sourceLen * targetLen
        if denominator == 0:
            return 0
        else:
            return numerator / denominator
SIM = Cosine()

class Cluster:
    def __init__(self):
        self.documents = []
        self.words = {}

    def updateCentre(self):
        centre = {}
        for document in self.documents:
            for word in document.keywords:
                if word not in centre:
                    centre[word] = 0
                centre[word] += 1
        self.words = dict([(word, centre[word] / len(self.documents)) for word in centre])

    def add(self, d):
        self.documents.append(d)
        self.updateCentre()

    def getCentre(self):
        return self.words


def link(score):
    return 1 - score < t


def nearest_neighbor_cluster_id(a, documents):
    tags = {}
    tagsa ={}
    for d in documents:
        if d.keywords not in tags:
            tags[d.keywords] = 0
        tags[d.keywords] += 1
    if a.keywords not in tagsa:
        tags[a.keywords] = 0
    tags[a.keywords] += 1
    if len(documents) > 0:
        score, b = sorted([(SIM.similar(tagsa, tags), d)], reverse=True)[0]
        if link(score):
            return b.clusterID
        else:
            return -1
    else:
        return -1


def LSH(documents):
    old_documents = []
    clusters = []
    for d in documents:
        cID = nearest_neighbor_cluster_id(d, old_documents)
        if cID == -1:
            cID = len(clusters)
            clusters.append(Cluster())

            d.clusterID = cID
            clusters[cID].add(d)
            old_documents.append(d)

        else:
            d.clusterID = cID
            clusters[cID].add(d)
            old_documents.append(d)

    # for c in clusters:
    #     print('cluster里document的个数',len(c.documents))

    eval_cluster(clusters)
    eval_cmin(clusters)


    return clusters