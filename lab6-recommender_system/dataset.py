import pandas as pd
import numpy as np
from tqdm import trange

def dataset():
    dfr=pd.read_csv('./ml-1m/ratings.dat',delimiter='::',header=None)
    dfr.columns=['UserID','MovieID','Rating','Timestamp']
    dfr.drop(columns=['Timestamp'])
    print(dfr.head())

    rating_matrix=dfr.pivot(index='UserID',columns='MovieID',values='Rating')
    n_users,n_movies=rating_matrix.shape

    # scaling ratings to between 0 and 1,this helps,our model by constraining predictions
    min_rating, max_rating = dfr['Rating'].min(), dfr['Rating'].max()
    rating_matrix =(rating_matrix-min_rating)/(max_rating-min_rating)

    # sparcity=rating_matrix.notna().sum().sum()/(n_users*n_movies)
    # print(f'Sparcity:{sparcity:0.2%}') 4.47%
    # print(f'rating matrix:{rating_matrix}')

    return rating_matrix

if __name__=='__main__':
    dataset()