import numpy as np
import pandas as pd
from tqdm import tqdm,trange

class LatentFactorModel:
    def __init__(self,n_users,n_items,n_factors,lr=0.01,reg=0.1,n_epochs=10):
        self.n_users=n_users
        self.n_items=n_items
        self.n_facotrs=n_factors
        self.lr=lr
        self.reg=reg
        self.n_epochs=n_epochs

        # initial
        self.user_factors=np.random.normal(scale=1.0/n_factors,size=(n_users,n_factors))
        self.item_factors=np.random.normal(scale=1.0/n_factors,size=(n_items,n_factors))

        # print(self.user_factors.shape,self.item_factors.shape)

    def fit(self,X:pd.DataFrame):
        for epoch in range(self.n_epochs):
            # suffle
            XShuffle = X.sample(frac=1).reset_index(drop=True)
            # update
            for i,row in tqdm(XShuffle.iterrows(),total=len(XShuffle)):
                user=row['userId']-1
                item=row['movieId']-1
                rating=row['rating']


                pred=np.dot(self.user_factors[user],self.item_factors[item])

                error=rating-pred

                self.user_factors[user]+=self.lr*(error*self.item_factors[item]-self.reg*self.user_factors[user])
                self.item_factors[item] += self.lr * (error * self.user_factors[user] - self.reg * self.item_factors[item])
            
    def predict(self,X:pd.DataFrame):
        preds=[]
        for i,row in tqdm(X.iterrows(),total=len(X)):
            user=row['userId']-1
            item=row['movieId']-1

            pred = np.dot(self.user_factors[user], self.item_factors[item])
            preds.append(pred)
        
        return np.array(preds)
    
    def evaluate(self,X):
        preds=self.predict(X)
        mse=np.mean((X['rating']-preds)**2)
        rmse=np.sqrt(mse)

        return rmse

users=pd.read_csv('./ml-1m/users.dat',delimiter="::",header=None)
items=pd.read_csv('./ml-1m/movies.dat',delimiter="::",header=None,encoding='latin-1')


ratings=pd.read_csv('./ml-1m/ratings.dat',delimiter="::",header=None)
ratings.columns=['userId','movieId','rating','timestamp']

ratings=ratings.drop(['timestamp'],axis=1)

train_data=ratings.sample(frac=0.8,random_state=1)
test_data=ratings.drop(train_data.index)

model=LatentFactorModel(n_users=users.iloc[:,0].max(skipna=False),n_items=items.iloc[:,0].max(skipna=False),n_factors=50)
model.fit(train_data)

print(f'rmse:{model.evaluate(test_data):.4f}')


