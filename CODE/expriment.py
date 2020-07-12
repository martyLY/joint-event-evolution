from my_class import *
from data_loader import *

def getExpTains(trains):
    X_train_list = []
    X_train_list1 = []
    X_train_time_list = []
    X_train_time_list1 = []

    y_train_story = []
    y_train_story1= []
    y_train_event = []
    y_train_event1 = []

    i = 0

    for d in trains:
        #story_id,event_id,content,keywords,main_keywords,time
        if i==0:
            X_train_list.append(d.title)
            X_train_time_list.append(d.time)
            y_train_story.append(d.story_id)
            y_train_event.append(d.event_id)
            i = 1
        else:
            X_train_list1.append(d.title)
            X_train_time_list1.append(d.time)
            y_train_story1.append(d.story_id)
            y_train_event1.append(d.event_id)
            i = 0

    # model = train_funtion([X_train_list,X_train_list1,X_train_time_list,X_train_time_list1], [y_train_story, y_train_story1,y_train_event, y_train_event1] ,
    #                       vocab_size) #vocab_size是输入的维度

    return [X_train_list,X_train_list1,X_train_time_list,X_train_time_list1], [y_train_story, y_train_story1,y_train_event, y_train_event1]