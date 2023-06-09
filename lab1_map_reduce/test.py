# import sys

# for line in sys.stdin:
#     print(line)

from itertools import groupby
from operator import itemgetter
from collections import Iterator

d1 = ['python', 100]
d2 = ['python', 99]
d3 = ['c++', 99]
d4 = ['c++',99]
d5 = ['python', 100]
d = [d1, d2, d3, d4, d5]
d.sort(key=lambda x:x[0], reverse = False )#分组之前先进行排序，改变了已经存在的列表，注意与sorted函数的区别
#排序后[['c++', 99], ['c++', 99], ['python', 100], ['python', 99], ['python', 100]]
lstg = groupby(d,lambda x:x[0])
for key, group in lstg:
    print (key,(isinstance(group,Iterator)))
    for g in group:         #group是一个迭代器
        print (g)