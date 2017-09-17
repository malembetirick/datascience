#!/usr/bin/python
#-*-coding:utf-8-*

import numpy

from plot import Generation
from TransformerMatrice import Transformer
from algorithm import MoindresCarres

X = numpy.loadtxt("x1.txt")
Y = numpy.loadtxt("y1.txt")

X = numpy.asmatrix(X)
Y = numpy.asmatrix(Y)

MSEDegreOpt = 10
MSEOpt = 10
CVDegreOpt = 10
CVOpt = 10

mses = []
cvs = []
rank = xrange(1, 15)
for i in rank :
    print("Generation de la courbe du polynome de rang : %d" % (i))
    name = "polynomeBase1_" + str(i)
    plot = Generation()

    plot.plotData(X, Y)

    transf = Transformer()
    Xraise = transf.raiseTo(X, i)

    algo = MoindresCarres(Xraise, Y.T, 300)
    theta = algo.evaluer_theta()

    tmp = numpy.arange(-5, 15, 0.1)
    res = numpy.zeros( tmp.shape[0] )
    for j in xrange(0, theta.shape[0]) :
        res += theta.item(j) * tmp ** (j)

    plot.plotCurveNamed(tmp, res, "rang " + str(i))
    plot.savefig("images/SurApprentissage1/"+name)
    plot.reset()
    print "...OK"

    mse = algo.error()
    cv = algo.validation_croisee(10)
    if mse < MSEOpt :
        MSEOpt = mse
        MSEDegreOpt = i

    if cv < CVOpt :
        CVOpt = cv
        CVDegreOpt = i

    print "MSE:", mse
    mses.append(mse)
    print "Validation croisee:", cv
    cvs.append(cv)
    print ""

print "MSE optimal:", MSEOpt
print "Degre optimal:", MSEDegreOpt
print "Validation croisee optimal:", CVOpt
print "Degre optimal:", CVDegreOpt
plot.plotData(rank, mses, "MSE")
plot.plotData(rank, cvs, "Cross Val", "r")
plot.savefig("images/SurApprentissage1/mse")
plot.reset()
