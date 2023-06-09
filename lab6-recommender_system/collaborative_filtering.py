import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ratings=pd.read_csv('./ml-1m/ratings.dat',delimiter="::",header=None)
ratings.columns=['userId','movieId','rating','timestamp']

ratings=ratings.drop(['timestamp'],axis=1)

train_data=ratings.sample(frac=0.8,random_state=1)
test_data=ratings.drop(train_data.index)

train_matrix = train_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
# print(train_matrix)

user_similarity=cosine_similarity(train_matrix)

def predict_user_based(user_similarity,train_matrix,user,item,k=10):
    # find k-nearnest
    similar_users=np.argsort(user_similarity[user])[::-1][1:k+1]
    # compute weight
    weights=user_similarity[user][similar_users]
    # find the score
    try:
        ratings=train_matrix.loc[similar_users+1, item]
        # compute predict score
        return np.dot(weights, ratings) / weights.sum()
    # if empty
    except:
        return train_matrix.loc[user].mean()

predictions = []
for i, row in tqdm(test_data.iterrows(), total=len(test_data)):
    user = row['userId']-1
    item = row['movieId']
    rating = row['rating']
    pred = predict_user_based(user_similarity, train_matrix, user, item)
    predictions.append(pred)
rmse = np.sqrt(np.mean((np.array(predictions) - test_data['rating']) ** 2))
mae = np.mean(np.abs(np.array(predictions) - test_data['rating']))
print('RMSE:', rmse)
print('MAE:', mae)
