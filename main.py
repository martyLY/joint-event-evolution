#coding=utf-8


from model import *
from data_loader import *
LEN_OF_TRAIN= 5000#20000 #200

data = getData() # docuemnt集合

# document的数量： 25941
tains = data[:LEN_OF_TRAIN]

tests = data[LEN_OF_TRAIN:7000]

print(tests)

JEDS(tains,tests)



