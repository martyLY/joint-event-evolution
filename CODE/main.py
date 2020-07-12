#coding=utf-8


from model import *
from data_loader import *
from test_modles import *
from jeds import *
from dyLDA import *
LEN_OF_TRAIN= 1100#1100 #20000 #200

data = getData() # docuemnt集合

# document的数量： 25941 2030
trains = data[:LEN_OF_TRAIN]

tests = data[LEN_OF_TRAIN:]


my_model(trains,tests)
#lsh_cluster(tests)
#JEDS(trains, tests)

