import xlrd
import csv
import codecs
import re

def xlsx_to_csv():
    workbook = xlrd.open_workbook("3.xlsx")
    table = workbook.sheet_by_index(0)
    t= []
    with codecs.open("3.csv", "w", encoding="utf-8") as f:
        wr = csv.writer(f)
        for row_num in range(table.nrows):
            row_value = table.row_values(row_num)
            wr.writerow(row_value)
                # story_id|event_id|title|content|category|timestamp|keywords|main_keywords
            # if(row_num>=1):
            #     story_id = str(int(row_value[0]))
            #
            #     event_id = str(int(row_value[1]))
            #     timestamp = str(int(row_value[2]))
            #     title = row_value[3]
            #     content = row_value[4]
            #     keywords = row_value[5]
            #
            #     main_keywords = row_value[6]
            #     category = str(int(row_value[7]))
            #     col = story_id + '|' + event_id + '|' + title + '|' + content + '|' + category + '|' + timestamp + '|' + keywords + '|' + main_keywords+'\n'
            #     print(col)
            #     t.append(col)

def csv_to_txt():
    input_file = open("3.csv")
    output_file = open("3.txt", "w")
    input_file.seek(0)
    header = input_file.readline()
    for line in input_file.readlines():
        documenttep = line.strip().split(',')
        story_id = documenttep[0][0:1]
        print(story_id)
        event_id = documenttep[1].rstrip('.0')
        print(event_id)
        tim = documenttep[2].rstrip('.0')
        title = documenttep[3]
        content = documenttep[4]
        if (len(documenttep) == 8):
            keywords = documenttep[5]
            main_keywords = documenttep[6]
            category = documenttep[7].rstrip('.0')
            # story_id|event_id|time|title|content|keywords|main_keywords|category
            #keywordstep = str(documenttep[6]).strip().split(',')
            # print(keywordstep)
            #main_keywordstep = str(documenttep[7]).strip().split(',')
            # story_id|event_id|title|content|category|timestamp|keywords|main_keywords
            col = story_id + '|' + event_id + '|' + title + '|' + content + '|' + category + '|' + tim + '|' + keywords + '|' + main_keywords + '\n'
            print(col)
            output_file.write(col)

        # print(main_keywordstep)
        # doc.append(document(documenttep[0], documenttep[1], documenttep[3],
        #                     keywordstep, main_keywordstep, documenttep[5]))

        input_file.seek(0)
    # for line in input_file:
    #     col = line.split('|')
    #     col[2] = int(col[2])
    #     table.append(col)
    # table_sorted = sorted(table, key=itemgetter(2))  # 先后按列索引3,4排序
    # #output_file.write(header)
    # for row in table_sorted:  # 遍历读取排序后的嵌套列表
    #     row[1]=""
    #     row = [str(x) for x in row]  # 转换为字符串格式，好写入文本
    #     output_file.write("|".join(row))
    # # for row in table_sorted:  # 遍历读取排序后的嵌套列表
    # #     row_list = []
    # #     j = random.randint(1,8)
    # #     for i in range(j):
    # #         row_list.append(row)
    # #     for row in row_list:
    # #         row = [str(x) for x in row]  # 转换为字符串格式，好写入文本
    # #         output_file.write("|".join(row))
    # print(i)
    input_file.close()
    output_file.close()

if __name__ == "__main__":
    #xlsx_to_csv()
    csv_to_txt()