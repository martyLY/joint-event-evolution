from __future__ import division
import math
from clster_layer import *
from my_class import *
import random


def eval_cmin(clusters):

    # C_min
    C_miss = 0.5
    C_FA = 0.5
    P_miss = 0
    P_FA = 0
    for i, c in enumerate(clusters):
        n = len(c.documents)
        fa = 0
        miss = 0

        tags = {}
        for d in c.documents:
            if d.event_id not in tags:
                tags[d.event_id]=0
            tags[d.event_id] += 1
            print(tags)

        tcount, tid = sorted([(count, event_id) for event_id, count in tags.items()], reverse=True)

        for d in c.documents:
            if d.event_id != tid:
                fa += 1

        for j in range(len(clusters)):
            if j != i:
                for d in clusters[j].documents:
                    if d.event_id == tid:
                        miss += 1
        fa = fa / len(c.documents)
        miss = miss / (miss + tcount)

        P_FA += fa
        P_miss += miss

    # print 'P_FA',P_FA/len(newClusters)
    # print 'P_Miss',P_miss/len(newClusters)
    print('C_min', C_FA * P_FA / len(clusters) + C_miss * P_miss / len(clusters))

def eval_cluster(clusters):
    cor_rata = 0
    cor_rata_list = []
    clusters = [c for c in clusters if len(c.documents) > 0]
    for i, c in enumerate(clusters):
        event_id_list = {}
        n = len(c.documents)
        for d in c.documents:
            if d.event_id not in event_id_list:
                event_id_list[d.event_id]=0
            event_id_list[d.event_id] += 1
        sorted(event_id_list.items(),key = lambda x:x[1],reverse = True)
        l = list(event_id_list.values())


        # print('出现次数最多的',l[0])
        # print(len(event_id_list))

        cor_rata = ((l[0]/len(event_id_list)) + cor_rata)/(i+1)
        cor_rata_list.append(cor_rata)
        #print('i', i,cor_rata)

    sorted(cor_rata_list,reverse = True)

    print('event的正确率：',cor_rata_list)

def eval_story(storise):
    cor_rata = 0
    cor_rata_list = []
    storise = [s for s in storise if len(s.cluters) > 0]
    for i,s in enumerate(storise):
        story_id_list = {}
        clusters = [c for c in s.cluters if len(c.documents) > 0]
        for c in clusters:
            for d in c.documents:
                if d.story_id not in story_id_list:
                    story_id_list[d.story_id] = 0
                story_id_list[d.story_id] += 1
            #print('',story_id_list)
        sorted(story_id_list.items(), key=lambda x: x[1], reverse=True)
        l = list(story_id_list.values())
        cor_rata = ((l[0] / len(story_id_list)) + cor_rata) / (i + 1)
        cor_rata_list.append(cor_rata)
        #print('i', i, cor_rata)


    sorted(cor_rata_list, reverse=True)
    print('story的正确率：', cor_rata_list)