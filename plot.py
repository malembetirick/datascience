#-*-coding:utf-8-*

import matplotlib.pyplot as pyplot


class Generation :

    def setLabel(self, xlabel, ylabel) :
        pyplot.xlabel(xlabel)
        pyplot.ylabel(ylabel)

    def plotData(self, X, Y, name="", color="b") :
        shape = pyplot.plot(X, Y, color+"o", label=name)

    def plotCurve(self, X, values, name="", color="g") :
        pyplot.plot(X, values, color, linewidth=2, label=name)

    def plotCurveNamed(self, X, values, name, color="g") :
        self.plotCurve(X, values, name, color)

    def plotPoint(self, x, y) :
        pyplot.plot(x, y, "ro")

    def reset(self) :
        pyplot.clf()

    def show(self) :
        pyplot.legend()
        pyplot.show()

    def savefig(self, filename) :
        pyplot.legend(scatterpoints=1)
        pyplot.savefig(filename)
