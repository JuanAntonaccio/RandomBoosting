# Arrancando hacer el pipeline del metodo
import matplotlib.pyplot as plt

import numpy as np 
import pandas as pd
import seaborn as sns
import pickle

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



print("Comenzando el proceso del Pipeline de RANDOM BOOSTING ....")

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv') 

df_transform=df_raw.drop(['Cabin','PassengerId','Ticket','Name'],axis=1)

# dos variables nuevas
df_transform['Sex_encoded']=df_transform['Sex'].apply(lambda x: 1 if x=="female" else 0)

df_transform = df_transform.drop(['Sex'],axis=1)

df_transform['Embarked_S']=df_transform['Embarked'].apply(lambda x: 1 if x=="S" else 0)

df_transform['Embarked_C']=df_transform['Embarked'].apply(lambda x: 1 if x=="C" else 0)

df_transform['Age_clean']=df_transform['Age'].fillna(30)

df_transform=df_transform.drop(['Embarked'],axis=1)
df_transform=df_transform.drop(['Age'],axis=1)

df=df_transform.copy()

X=df.drop(['Survived'],axis=1)

y=df['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=70)


pickle.dump(cl, open('../models/random_forest.pkl', 'wb'))

print()
print("="*80)
print(" Se guardo el modelo armado solicitado para poder trabajar con el en la carpeta models ")
print()
print("                F i n     d e l    P r o g r a m a")
print()
print("="*80)
