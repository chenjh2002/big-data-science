import numpy as np
import pandas as pd
import re
import time
from datasketch import MinHash,MinHashLSHForest

#Preporcess wiil split a tsring pf test into indiviual tokens/singles
def preprocess(text):
    text=re.sub(r'[^\w\s]','',text)
    token=text.lower()
    tokens=token.split()
    return tokens

#Number of Permutation
permutations=128

#Number of Recommendation to return
num_recommendations=10

def get_forest(data,perms):
    start_time=time.time()
    
    minhash=[]
    
    for text in data['text']:
        tokens=preprocess(text)
        m=MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf-8'))
        minhash.append(m)
    
    forest=MinHashLSHForest(num_perm=perms)
    
    for i,m in enumerate(minhash):
        forest.add(i,m)
    
    forest.index()
    
    print('It took %s seconds to build forest.' %(time.time()-start_time))

    return forest

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

if __name__=='__main__':
    db=pd.read_csv('papers.csv')
    db['text']=db['title']+' '+db['abstract']
    forest=get_forest(db,permutations)
    
    title = 'Using a neural net to instantiate a deformable model'
    # title="Self-Organization of Associative Database and Its Application"
    result = predict(title, db, permutations, num_recommendations, forest)
    print('\n Top Recommendation(s) is(are) \n', result)
