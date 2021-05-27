from pyforest import *

df = pd.read_csv("Zomato_df.csv")
print(df.head())
df.drop('Unnamed: 0',axis =1,inplace=True)
print(df.head())
X = df.drop('rate',axis=1)
y = df['rate']
print("Splitting.....")
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=123)

print("Model building.....")
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor()

model_rf.fit(X_train,y_train)

y_pred = model_rf.predict(X_test)

print(y_pred)

import pickle 

pickle.dump(model_rf,open('model_rf.pkl','wb'))

print("Pickle file created.....")

