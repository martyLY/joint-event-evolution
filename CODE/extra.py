from time import mktime
import time
import random
def w_file():
    fr = open('1_lable.txt')
    doc = []
    print('start load data...')
    fr.seek(0)
    for line in fr.readlines():
        col = line.split('|')

def find():
    input_file = open("CNESC.txt")
    input_file.seek(0)
    story_id_pre = 0
    table = {}
    i = 0
    for line in input_file.readlines():
        documenttep = line.strip().split('|')
        if(documenttep[0]!=''):
            story_id = int(documenttep[0])
            if(story_id!=story_id_pre):
                story_id_pre = story_id
                i = 0
            else:
                i +=1
                table.update({story_id:i})
    sorted(table.items(), key=lambda x: x[0], reverse=True)
    print(table)

def jiakong():
    input_file = open("stories.txt")
    output_file = open("storise*.txt", "w")
    input_file.seek(0)
    story_id_pre = 0
    table = []
    i = 0
    for line in input_file.readlines():
        if(line[0]=='|'):
            table.append(line)
            print(line)
            output_file.write(line+'\n')

def creadata():
    output_file = open("s.csv", "w")
    for o in range(50):
        i = random.randint(5,18)
        #i = random.uniform(75,85)
        output_file.write(str(i))
        output_file.write('\n')
        print(i)




if __name__ == "__main__":
    #xlsx_to_csv()
    #find()
    #jiakong()
    creadata()
# input_file = open("000.txt")
# output_file = open("000_lable.txt", "w")
# output_file.write('\n')
# i =1
# j=0
# table = []
# #header = input_file.readline()  # 读取并弹出第一行
# input_file.seek(0)
# for line in input_file.readlines():
#     documenttep = line.strip().split('|')
#     story_id = documenttep[0]
#     event_id = documenttep[1]
#     tim = documenttep[2]
#     if(story_id=='11' or story_id=='12'):
#         timeArray = time.strptime(tim , "%Y-%m-%d %H:%M")
#         tim = str(mktime(timeArray)) #2018-07-14 08:08
#     title =documenttep[3]
#     content = documenttep[4]
#     if(len(documenttep) == 8):
#         keywords = documenttep[5]
#         main_keywords = documenttep[6]
#         category = documenttep[7]
#         # story_id|event_id|time|title|content|keywords|main_keywords|category
#         keywordstep = str(documenttep[6]).strip().split(',')
#         #print(keywordstep)
#         main_keywordstep = str(documenttep[7]).strip().split(',')
#         # story_id|event_id|title|content|category|timestamp|keywords|main_keywords
#         col = story_id+'|'+event_id+'|'+title+'|'+content+'|'+category+'|'+tim+'|'+keywords+'|'+main_keywords+'\n'
#         print(col)
#         output_file.write(col)
#
#     #print(main_keywordstep)
#     # doc.append(document(documenttep[0], documenttep[1], documenttep[3],
#     #                     keywordstep, main_keywordstep, documenttep[5]))
#
#     input_file.seek(0)
# # for line in input_file:
# #     col = line.split('|')
# #     col[2] = int(col[2])
# #     table.append(col)
# # table_sorted = sorted(table, key=itemgetter(2))  # 先后按列索引3,4排序
# # #output_file.write(header)
# # for row in table_sorted:  # 遍历读取排序后的嵌套列表
# #     row[1]=""
# #     row = [str(x) for x in row]  # 转换为字符串格式，好写入文本
# #     output_file.write("|".join(row))
# # # for row in table_sorted:  # 遍历读取排序后的嵌套列表
# # #     row_list = []
# # #     j = random.randint(1,8)
# # #     for i in range(j):
# # #         row_list.append(row)
# # #     for row in row_list:
# # #         row = [str(x) for x in row]  # 转换为字符串格式，好写入文本
# # #         output_file.write("|".join(row))
# # print(i)
# input_file.close()
# output_file.close()


