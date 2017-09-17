#-*-coding:utf-8-*
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import sklearn.cross_validation as cv
from sklearn.datasets import make_circles

def phi(x):
    return np.vstack(np.power(x,2))

X, Y = datasets.make_circles(n_samples=500,noise=0.05,
factor=0.5, random_state=0)

XX = phi(X)
#print(XX)

X_train, X_test, Y_train, Y_test = cv.train_test_split(X, Y, test_size=0.2, train_size=0.8)

#XX = phi(X_train,X_test)

fig, axes = plt.subplots(nrows=1, ncols=1)
axes.scatter(X[:,0], X[:,1], c=Y)

#clf = svm.SVC(kernel='poly',degree=2,C=1.0)
clf = svm.SVC(kernel='linear',C=0.001,
shrinking=False, probability=False,
max_iter=500)

clf.fit(XX,Y)

hY = clf.predict(X_train)
print "Nombre d'erreurs:", np.mean(abs(hY - Y_train)), 'sur la base d apprentissage', len(Y_train)

plt.show()
