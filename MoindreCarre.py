# -*- coding: utf8 -*-

import numpy
from plot import Generation

def load_data():
  """
	Charge les donnees des fichiers textes.
  """
  global x, y, N, t, p
  t = numpy.loadtxt('t.txt')
  N = len(t)
  x = numpy.vstack((t, numpy.ones(N)))
  p = numpy.loadtxt('p.txt')
  y = p


def print_results():
  """
	Affiche les résultats.
  """
  print 'FAA - TP2: Moindres carres'
  print '-' * 80
  print 'Nombre de donnees: {0}'.format(N)
  print '-' * 80
  print 'Théta: {0}'.format(theta())
  print '-' * 80
  print 'Erreur quadratique: {0}'.format(j_theta())

def theta():
  """
	Calcul théta par la méthode des moindres carrés pour x et y.
  """
  return numpy.dot(numpy.linalg.inv(numpy.dot(x, x.T)), numpy.dot(x, y))

def f_theta():
  """
	Calcul le y pour les x en fonction de théta.
  """
  return numpy.dot(theta().T, x)

def j_theta():
  """
	Calcul de l'erreur quadratique.
  """
  tmp = (y - numpy.dot(x.T, theta()))
  return ((1.0/N) * numpy.dot(tmp.T, tmp))

def print_graphs():
  """
	Affiche les données sur le graphe.
  """
  print "Generation d'un graphe..."
  plot = Generation()
  plot.plotData(t, p)
  plot.plotCurveNamed(t.T, (numpy.dot(x.T,theta())), "")
  plot.setLabel("x = temps", "y = position")
  plot.show()
  plot.savefig("MoindresCarres.png")
  print "Graph fait!"

def main():
  """
	Fonction principale du programme.
  """
  load_data()
  print_results()
  print_graphs()

if __name__ == '__main__':
  main()
