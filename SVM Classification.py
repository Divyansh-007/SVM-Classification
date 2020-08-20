# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:36:50 2020

@author: Dell
"""

# Importing all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Here we will be using the popular iris flower dataset
# So lets first see the images of three different iris types

# Iris Setosa
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)

# Iris Versicolor
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)

# Iris Virgininca
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)

# Importing the dataset
iris = sns.load_dataset('iris')

# Spliting the dataset into training and testing 
X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Training the SVM model
svc_model = SVC()
svc_model.fit(X_train,y_train)

# Testing and Evaluating the SVM model
svc_predictions = svc_model.predict(X_test)
print(confusion_matrix(y_test, svc_predictions))
print(classification_report(y_test, svc_predictions))

# Tuning the parameters using Grid Search and refiting the model
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)

# Getting the best parameters
grid.best_params_
grid.best_estimator_

# Predicting and Evaluating again with the use of new parameters
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions)) 