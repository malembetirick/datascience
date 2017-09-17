# -*- coding: utf8 -*-

import numpy
from matplotlib import pyplot

def load_data():
  """
	Charge les données des fichiers textes.
  """
  global f, h, f_t, h_t, f_p, h_p, f_h_t, f_h_p, N, x_t, y_t, x_p, y_p
  f = numpy.loadtxt('taillepoids_f.txt')
  h = numpy.loadtxt('taillepoids_h.txt')

  f_t = f.copy()
  h_t = h.copy()
  f_t[:,0] = numpy.zeros(len(f_t))
  h_t[:,0] = numpy.ones(len(h_t))
  f_h_t = numpy.concatenate((f_t, h_t), axis=0)

  f_p = f.copy()
  h_p = h.copy()
  f_p[:,1] = numpy.zeros(len(f_p))
  h_p[:,1] = numpy.ones(len(h_p))
  f_h_p = numpy.concatenate((f_p, h_p), axis=0)

  N = len(f_h_t)

  x_t = numpy.vstack((f_h_t[:,0], numpy.ones(N)))
  y_t = f_h_t[:,1]


  x_p = numpy.vstack((f_h_p[:,0], numpy.ones(N)))
  y_p = f_h_p[:,1]

def sigmoid_taille(x_t) :
	return 1.0 / (1.0 + numpy.exp(-1.0 *x_t))

def sigmoid_poids(x_p,a,b=0.0) :
	return 1.0 / (1.0 + numpy.exp(-1.0 *(x_p)))

def compute_cost_taille(x_t, y_t, theta_t):#cout pour la dimension taille
	p_1 = sigmoid_taille(numpy.dot(theta_t.T,x_t)) # prediction proba appartenance classe 1
	log_l = (numpy.dot(-y_t,numpy.log(p_1)) - numpy.dot(1-y_t,numpy.log(1-p_1)))/x_t.shape[1] # log de vraisemblance
	return log_l

def compute_cost_poids(x_p, y_p, theta_p):#cout pour la dimension poids
	p_1 = sigmoid_poids(numpy.dot(theta.T,x_p)) # prediction proba appartenance classe 1
	log_l = (1./x_p.shape[1])*(-(y_p).dot(numpy.log(p_1)) - (1-y_p).dot(numpy.log(1-p_1))) # log de vraisemblance
	return log_l

def gradient_descent_stochastique_taille(alpha,x_t, y_t, num_iters):
	costs = numpy.zeros(num_iters)
	i=0
	while i < num_iters:
		for j in range(theta_t.shape[1]):
			hx = sigmoid_taille(numpy.dot(theta_t.T, x_t))
			diff = y_t-hx
			theta_t[j] = theta_t[j] + (alpha) * numpy.dot(diff, x_t[j, :].T)
			i+=1
			costs[i-1] = compute_cost_taille(x_t, y_t, theta_t)
			print "iter regression_logistique sur la dimension taille %s | risque empirique: %.3f" % (i, costs[i-1])
	return theta, costs


def print_graphs():
  """
	Affiche les données sur le graph.
  """
  print 'Légende figure 2 et 3:'
  print '	Les points affichés sur les sigmoides sont des projections des x sur celles-ci.'
  print '	Si le point est au dessus de la courbe de tau, il sera placé dans la catégorie des hommes.'
  print '	Si le point est en dessous de la courbe de tau, il sera placé dans la catégorie des femmes.'
  print '-' * 80

  pyplot.figure(1)
  pyplot.plot(f[:,0], f[:,1], '.', label="femme")
  pyplot.plot(h[:,0], h[:,1], '.', label="homme")
  pyplot.title('donnees')
  pyplot.grid(True)
  pyplot.legend()

  pyplot.figure(2)
  pyplot.plot(f_t[:,0], f_t[:,1], '.', label="f_t")
  pyplot.plot(h_t[:,0], h_t[:,1], '.', label="h_t")
  pyplot.title('Tailles')
  pyplot.grid(True)
  pyplot.plot(f_h_t[:,0], (1 - sigmoid_taille(numpy.dot(theta_t.T,x_t))), '.', label="sigmoide")

  pyplot.plot(h_t[51][0], (1 - sigmoid(numpy.dot(theta_t.T,numpy.array([h_t[51][0], 1])))), 'o g')
  pyplot.plot(h_t[875][0], (1 - sigmoid(numpy.dot(theta_t.T, numpy.array([h_t[875][0], 1])))), 'o g')

  pyplot.plot(f_t[51][0], (1 - sigmoid(numpy.dot(theta_t.T, numpy.array([f_t[51][0], 1]).T))), 'o b')
  pyplot.plot(f_t[875][0], (1 - sigmoid(numpy.dot(theta_t.T, numpy.array([f_t[875][0], 1])))), 'o b')

  tmp = []
  for i in f_h_t[:,0]:
	tmp.append(tau_t)

  pyplot.plot(f_h_t[:,0], tmp, label="seuil")
  pyplot.legend()

  pyplot.show()

def perf_taille(seuil):
  """
	Calcul la performance de la classification (taux d'erreur) pour la dimension des tailles.
  """
  res = []
  for i in range(0, N):
	if ((1 - sigmoid(numpy.dot(theta_t.T, numpy.array([f_h_t[i][0], 1])))) < seuil):
	  res.append(0)
	else:
	  res.append(1)

  sum = 0
  for i in range(0, N):
	sum += abs(f_h_t[i][1] - res[i])

  return (1.0/N) * sum

def main():
  """
	Fonction principale du programme.
  """
  global theta_t, theta_p, tau_t, tau_p
  load_data()
  theta_t = numpy.zeros((2, 1))
  iterations = 10000
  alpha = 0.0001

  print 'FAA - TP5: Régression logistique'
  print '-' * 80
  print 'Nombre de données: {0}'.format(N)
  print '-' * 80

  theta_t,costs = gradient_descent_stochastique_taille(alpha,x_t, y_t, iterations)
  print "theta regression_logistique",theta_t.T

  tau_t = 0.99
  print 'Calcul de la performance pour la dimension des tailles:'
  print '	- Valeur de tau: {0}'.format(tau_t)
  print '\n	Performance: {0}'.format(perf_taille(tau_t))
  print '-' * 80

  print_graphs()

if __name__ == '__main__':
  main()
