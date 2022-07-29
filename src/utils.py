import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier

url = 'https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'

df = pd.read_csv(url)

df = df.drop(['PassengerId','Cabin', 'Ticket', 'Name'], axis = 1)

df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

embarked_dict = {'S': 0, 'C': 1, 'Q':2}
df['Embarked'] = df['Embarked'].map(embarked_dict)

df['Survived']=pd.Categorical(df['Survived'])
df['Sex']=pd.Categorical(df['Sex'])
df['Embarked']=pd.Categorical(df['Embarked'])

df_processed = df.copy()
df_processed['Age'].fillna(df_processed['Age'].mean(), inplace = True)
df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace = True)

X = df_processed.drop(['Survived'], axis=1)
y = df_processed['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

D_train = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)

parametros = {
    'eta': 0.2,     # 0.3 por defecto
    'max_depth': 4,  # 6 por defecto
    'objective': 'multi:softmax',  
    'num_class': 2} 

steps = 20

XGB = xgb.train(parametros, D_train, steps)

# Guardar modelo
filename = '../models/modelo_boosting.sav'
pickle.dump(XGB, open(filename, 'wb'))

print('Se guardó el modelo')