#-*-coding:utf-8-*
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import sklearn.cross_validation as cv
from sklearn.datasets.samples_generator import make_blobs

X, Y = datasets.make_blobs(centers=2,
n_samples=500, cluster_std=0.3, random_state=0)

X_train, X_test, Y_train, Y_test = cv.train_test_split(X, Y, test_size=0.2, train_size=0.8)

fig, axes = plt.subplots(nrows=1, ncols=1)
axes.scatter(X[:,0], X[:,1], c=Y)

#C represente le coefficient de regularisation
clf = svm.SVC(kernel='linear',C=1.0,
shrinking=False, probability=False,
max_iter=500)

clf.fit(X,Y)

#frontière decision
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - clf.intercept_[0] / w[1]

#marges et vecteurs supports
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

h0 = plt.plot(xx, yy, 'k-', label='frontiere decision')
h1 = plt.plot(xx, yy_down, 'k--', label='marge1')
h2 = plt.plot(xx, yy_up, 'k--', label='marge2')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
s=80, facecolors='none')

hY = clf.predict(X_train)
print "Nombre d'erreurs:", np.mean(abs(hY - Y_train)), 'sur la base d apprentissage', len(Y_train)

plt.show()
