# breast cancer classification
#importing modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#getting dataset
dataset= pd.read_csv("data.csv")
X= dataset.iloc[:,2:32].values
y= dataset.iloc[:,1].values

#Categorical values to integers.
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

#spliting train-test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
Sc_X= StandardScaler()
X_train= Sc_X.fit_transform(X_train)
X_test= Sc_X.transform(X_test)

#Pca
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train= pca.fit_transform(X_train)
X_test= pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#building SVM Classifier
from sklearn.svm import SVC
classifier= SVC(kernel="linear", random_state=0)
classifier.fit(X_train,y_train)


#predicting variables
y_pred= classifier.predict(X_train)
y_predt= classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_train,y_pred)
cms= confusion_matrix(y_test,y_predt)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Component1')
plt.ylabel('Component2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Component1')
plt.ylabel('Component2')
plt.legend()
plt.show()

#Visualisation Correlation matrix
df= pd.read_csv("brcancer.csv")
matrix=df.corr(method ='kendall') 
