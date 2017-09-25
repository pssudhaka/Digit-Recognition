import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

data= pd.read_csv("train.csv")

data.head()

X = data.iloc[:,1:]
Y= data.iloc[:,0]

X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state = 0)

rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)

y_predict=rfc.predict(X_test)

rfc.score(X_test,y_test)
rfc.score(X_train,y_train)

cv_score=cross_val_score(rfc,X,Y,cv=5)

cv_score

np.mean(cv_score)