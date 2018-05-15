# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 09:44:00 2018

@author: dell 1
"""

#classification template
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import dataset
dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,2:4].values
Y=dataset.iloc[:,4].values
"""#takecare of missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values=('NaN'),strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
#encoding categirical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()
Y=labelencoder_y.fit_transform(Y)"""
#splitting the training and test set
from sklearn.cross_validation import train_test_split
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.25,random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
train_X=sc_X.fit_transform(train_X)
test_X=sc_X.transform(test_X)
#fitting the clssifier into training_set
from sklearn.neighbors import KNeighborClassifier
classifier=#classification template
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
#fitting the k-nn clssifier into training_set
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
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
plt.title('K-NN (Training set)')
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
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()





