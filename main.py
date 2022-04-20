import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Startups.csv')

print(data)

states = data.iloc[:,3:4].values

# Categorization of Data
from sklearn import preprocessing 

le = preprocessing.LabelEncoder()

states[:, 0] = le.fit_transform(states[:,0])
print(states)

# Creating dataframe from array and combining
states_df = pd.DataFrame(data = states, index = range(len(states)), columns = ["States"])

data_e = pd.concat([data.iloc[:,:3], states_df], axis=1)
data_e = pd.concat([data_e, data.iloc[:,4:]], axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_e.iloc[:,:4], data_e.iloc[:,4:], test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)