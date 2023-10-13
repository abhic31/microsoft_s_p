import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib



data = pd.read_csv("MSFT.csv")
print(data.head())


#dividing train and test data
x = data[["Open", "High", "Low"]]
y = data["Close"]
x = x.to_numpy()# convert to numpy array
y = y.to_numpy()
y = y.reshape(-1, 1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

#TODO: used decision tree regressor to predict the closing price of the stock

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
data = pd.DataFrame(data={"Predicted Rate": ypred})
print(data.head())

#saving the model in joblib
joblib.dump(model, 'model.pkl')