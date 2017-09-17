#!/usr/bin/python
#-*-coding:utf-8-*

import numpy

class Transformer :

    def raiseTo(self, vector, n) :
        """
        Cree une nouvelle matrice ou chaque ligne est egale au vecteur puissance l'indice de la ligne.
        Le nombre de ligne varie de 0 à n
        """
        if n < 1 :
            return numpy.power(vector, 0)
        else :
            above = self.raiseTo(vector, n-1)
            line = numpy.power(vector, n)
            return numpy.vstack( (above, line) )
