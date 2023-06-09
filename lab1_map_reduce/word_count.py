from functools import reduce
import numpy as np
from typing import List,Dict
from collections import defaultdict

def generate_random_words_file(sample_num:int=100):
    
    with open("word.txt","w") as file:
        random_words=np.random.randint(97,110,(3,sample_num))
        for row in range(sample_num):
            words="".join([chr(each) for each in random_words[:,row]])
            
            file.write(words+'\n')

def read_file_by_chunk(lines:int =10)->list:
    res=[]
    with open("word.txt","r") as file:
        tmp_chunk=[]
        for idx,line in enumerate(file.readlines()):
            tmp_chunk.append(line.strip())
            if idx%lines ==0:
                res.append(tmp_chunk)
                tmp_chunk=[]
    return res

def map_count(data:List[str])->Dict[str,int]:
    word_count=defaultdict(int)
    for item in data:
        word_count[item]+=1
    
    return word_count

def reduce_count(data1:Dict[str,int],data2:Dict[str,int])->Dict[str,int]:    
    for k,v in data2.items():
        data1[k]=data1[k]+data2[k]
    
    return data1

if __name__ == '__main__':
    generate_random_words_file(sample_num=100000)
    data_chunk = read_file_by_chunk(lines=100)  # 数据切片
    map_res = map(map_count, data_chunk)  # map
    reduce_res = reduce(reduce_count, map_res)  # reduce
    reduce_res = sorted(reduce_res.items(), key=lambda x: x[1], reverse=True)  # 排序

    for each in reduce_res[:10]:
        print(each)


