# 大数据原理与技术实验二
学号:20337011 姓名：陈俊皓

## 实验目标
实现一个local sensitive hashing(LSH)算法，以计算与给定论文标题最为相似的论文集。

## 实验过程

### 数据读取过程
首先是用`python`的 `panadas`库读取`.csv`中的数据，其中包含了论文的基础信息(标题+摘要)。此处的数据集来自于`NIPS`在`Kaggle`共享的论文信息数据。实现的代码如下:
```python
    db=pd.read_csv('papers.csv')
    db['text']=db['title']+' '+db['abstract']
```

### Singling 文本转换为tokens
在读取到每个文本的数据之后，需要将其先转换为`tokens`以方便后续取样及计算相似度。实现的代码如下:
```python
def preprocess(text):
    text=re.sub(r'[^\w\s]','',text)
    token=text.lower()
    tokens=token.split()
    return tokens
```
### MinHashing
随后，我们利用`python`中`datasketch`库提供的`MinHash`方法，对原来的`tokens`集合进行重排，从而降低空闲损耗。实现的代码如下:
```Python
    for text in data['text']:
        tokens=preprocess(text)
        m=MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf-8'))
        minhash.append(m)
```

### LSH
LSH利用对得到的签名(signature)进行划分并hash的方法，来计算不同*列*之间的相似性，具体的流程如下:
1. 将签名矩阵(`signature matrix`)划分成`b`个带，每个带有`r`行。
2. 对于每个带，通过一个hash函数将其中的每一列映入一个bucket中(共k个bucket)。
3. 对于被映入同一个bucket的列，候选他们为相似的数据特征。
4. 微调`b`和`r`找到最相似的对。

利用`MinHashLSHForest`构造hash函数的过程如下:
```Python
forest=MinHashLSHForest(num_perm=perms)
    
    for i,m in enumerate(minhash):
        forest.add(i,m)
    
    forest.index()
```

### 预测过程
以下是实现的预测过程代码:
```Python
def predict(text,database,perms,num_results,forest):
    start_time=time.time()
    
    tokens=preprocess(text)
    
    m=MinHash(num_perm=perms)
    for s in tokens:
        m.update(s.encode('utf-8'))
        
    
    idx_array=np.array(forest.query(m,num_results))
    if len(idx_array)==0:
        return None
    
    result=database.iloc[idx_array]['title']
    
    print('It took %s seconds to query forest.' %(time.time()-start_time))
    
    return result
```

## 实验结果



