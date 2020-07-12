from __future__ import division
import math
from cluster_layer import *
from my_class import *
import random

def readResult(results):
    #P = 真正预测准确的数量 / 预测是准确的数量
    #R = 真正预测准确的数量 / 所有真正好的数量
    i = 1
    index=0
    p=n=tp=tn=fp=fn=0
    for prob in results:
        #print(prob)
        index +=prob
        if prob>0.5:
            predLabel=1
        else:
            predLabel=0
        # if i>0:
        p += 1
        if predLabel == 1:
            tp += 1
            fp += 1
        else:
            fn += 1
            tn += 1
        # else:
        #     n += 1
        #     # if predLabel < 0:
        #     #     tn += 1
        #     # else:
        #     fp += 1
        #     i=1

    # if(tp==0 and fp==0):
    #     tp=0.5

    acc=(tp+tn)/(p+n)
    precisionP=tp/(tp+fp)
    #precisionP = index
    recallP=tp/(tp+fn)
    # if (precisionP == 0 and recallP == 0):
    #     precisionP = 0.5
    f_p=2*precisionP*recallP/(precisionP+recallP)
    #f_n=2*precisionN*recallN/(precisionN+recallN)
    print('precision:',precisionP)
    print('recall:',recallP)
    print ('{f1:%s} ' %(f_p))
    #print('AUC %s' %average_precision_score(y_test,results))

    #output=open('result.output','w')
    #output.write('\n'.join(['%s' %r for r in results]))

def eval_cmin(clusters):

    # C_min
    C_miss = 0.5
    C_FA = 0.5
    P_miss = 0
    P_FA = 0
    clusters = [c for c in clusters if len(c.documents) > 0]
    for i, c in enumerate(clusters):
        n = len(c.documents)
        fa = 0
        miss = 0

        tags = {}
        for d in c.documents:
            if d.event_id not in tags:
                tags[d.event_id]=0
            tags[d.event_id] += 1
            #print(tags)

        #print(sorted([ count for count in tags.values()], reverse=True))
        tcount = list(sorted([ count for count in tags.values()], reverse=True))[0]
        tid = list(tags.keys())[0]
        # tcount0 = sorted([count for count in tags.values()], reverse=True)
        # tcount = tcount0[0]
        # tid0 = sorted([id for id in tags.keys()], reverse=True)
        # tid = tid0[0]

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
    cor_sum = 0
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


        cor_rata = ((l[0]/len(event_id_list)) + cor_rata)/(i+1)
        cor_rata_list.append(cor_rata)
        #print('i', i,cor_rata)

    sorted(cor_rata_list,reverse = True)
    for cor in cor_rata_list[:4]:
        cor_sum +=cor
    re = cor_sum/len(cor_rata_list[:4])

    print('event的正确率：',re)

def eval_story(storise):
    cor_rata = 0
    cor_sum = 0
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
    for cor in cor_rata_list[:4]:
        cor_sum +=cor
    re = cor_sum/len(cor_rata_list[:4])
    print('story的正确率：',re)