# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:00:43 2018

@author: dell 1
"""
#svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import dataset
dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,2:4].values
Y=dataset.iloc[:,4].values

#splitting the training and test set
from sklearn.cross_validation import train_test_split
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.25,random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
train_X=sc_X.fit_transform(train_X)
test_X=sc_X.transform(test_X)
#fitting the clssifier into training_set
from sklearn.svm import SVC
classifier= SVC(kernel= 'linear',random_state=0)
classifier.fit(train_X,train_Y)

#predicting the test set result
Y_pred=classifier.predict(test_X)
#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(test_Y,Y_pred)
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = train_X,train_Y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('svm (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = test_X, test_Y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

