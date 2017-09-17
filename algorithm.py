#-*-coding:utf-8-*

import numpy

class Algorithm(object) :

    def __init__(self, X, Y, size) :
        print("Initialisation de l'algo...")
        self.matrix1 = X
        self.matrix2 = Y
        self.size = size
        self.theta = None

    def evaluer_theta(self) :
        raise NotImplementedError()

    def error(self) :
        raise NotImplementedError()

    def mse(self,k, start, end) :
        cost = self.matrix2[range(start, end),:] - numpy.dot(self.matrix1[:,range(start, end)].T, self.theta)
        cost = pow(numpy.linalg.norm(cost), 2)
        mean = (1.0 * k / self.size) * cost
        return mean

    def validation_croisee(self, k) :
        risk = 0.0
        part = self.size / k
        for i in range(k) :
            mse = self.mse(k,i * part, (i * part) + part)
            risk += (1.0 * k / self.size) * mse
        return (1.0 / k) * risk

class MoindresCarres(Algorithm) :

    def __init__(self, X, Y, size) :
        super(MoindresCarres, self).__init__(X, Y, size)

    def evaluer_theta(self) :
        print("Calcul de theta...")
        part1 = numpy.dot(self.matrix1, self.matrix1.T).I
        part2 = numpy.dot(self.matrix1, self.matrix2)
        self.theta = numpy.dot(part1, part2)
        return self.theta

    def error(self) :
        return self.mse(1,0, self.size)
