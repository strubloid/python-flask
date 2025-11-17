## this data comes from : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

## load the data
df = pd.read_csv('diabetes.csv')

## split the data into features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

## split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## train a KNN model
knn = KNeighborsClassifier()

## fit the model
knn.fit(X_train, y_train)

## evaluate the model
# knn_score = knn.score(X_test, y_test)

## save the model to disk
pickle.dump(knn, open('example_weights_knn.pkl', 'wb'))