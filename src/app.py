# Arrancando hacer el pipeline del metodo
import matplotlib.pyplot as plt

import numpy as np 
import pandas as pd
import seaborn as sns
import pickle
import xgboost as xgb

from xgboost import XGBClassifier
from sklearn.ensemble import  RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from urllib.request import urlretrieve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV



print("Comenzando el proceso del Pipeline de X BOOSTING ....")

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



D_train = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
D_test = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

evaluacion = [(D_test, "eval"), (D_train, "train")]

def metricas(objetivo, prediccion):
    matriz_conf = confusion_matrix(objetivo, prediccion)
    score = accuracy_score(objetivo, prediccion)
    reporte = classification_report(objetivo, prediccion)
    metricas = [matriz_conf, score, reporte]
    return(metricas)

parametros_02 = {"booster":"gbtree", "max_depth": 4, "eta": .3, "objective": "binary:logistic", "nthread":2}
rondas_02 = 100

modelo_02 = xgb.train(parametros_02, D_test, rondas_02, evaluacion, early_stopping_rounds=10)

prediccion_02 = modelo_02.predict(D_test)
prediccion_02 = [1 if i > .7 else 0 for i in prediccion_02]

metricas_02 = metricas(y_test, prediccion_02)

[print(i) for i in metricas_02]

# Guardar modelo
filename = '../models/modelo_boosting.sav'
pickle.dump(modelo_02, open(filename, 'wb'))

print()
print("="*80)
print(" Se guardo el modelo armado solicitado para poder trabajar con el en la carpeta models ")
print()
print("                F i n     d e l    P r o g r a m a")
print()
print("="*80)
