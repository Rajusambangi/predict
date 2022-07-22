import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
df=pd.read_csv('iris.csv')
#df.head()
#df.isna().sum()
from sklearn.linear_model import LogisticRegression
x=df.drop(['Id','Species'],axis=1)
y=df['Species']
lr=LogisticRegression(class_weight='balanced')
lr.fit(x,y)
y_pred=lr.predict(x)
from sklearn.metrics import accuracy_score
accuracy_score(y,y_pred)
with open('iris.pkl','wb') as f:
    pickle.dump(lr,f)
df['Species'].value_counts()
lr_model = pickle.load(open('iris.pkl','rb'))
lr_model.predict([[5.1,3.5,1.4,0.2]])[0]