from cluster_layer import *
from fir_layer import *
from story_layer import *
from operator import itemgetter

def out_txt_story(storise):

    fr = open('output_story.txt','w')
    fr.write('story_id|event_id|time|title|keyword'+ '\n')
    for s in storise:
        for c in s.cluters:
            for d in c.documents:
                fr.write(str(d.story_id) + '|' + str(d.event_id) + '|'+ str(d.time)+ '|'+  str(d.title) + '|' + str(
                    d.get_keywords()) + '|' +'\n')

    fr.close()

def out_sort_story():
    input_file = open("output_story.txt")
    output_file = open("output_sort_story.txt", "w")

    table = []
    header = input_file.readline()  # 读取并弹出第一行
    for line in input_file:
        col = line.split('|')
        col[0] = int(col[0])
        col[1] = int(col[1])
        #col[2] = int(col[2].rstrip('.0'))
        table.append(col)
    #table_sorted = sorted(table, key=itemgetter(0, 4))  # 先后按列索引3,4排序
    table_sorted = sorted(table, key=itemgetter(0,1))
    #output_file.write(header + '\t')
    for row in table_sorted:  # 遍历读取排序后的嵌套列表
        row = [str(x) for x in row]  # 转换为字符串格式，好写入文本
        output_file.write("|".join(row) + '\n')

    input_file.close()
    output_file.close()






