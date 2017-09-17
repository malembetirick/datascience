import numpy as np
import random
import pylab
from plot import Generation
from sklearn.preprocessing import normalize

global x, y, N, t, p
t = np.loadtxt('t.txt')
N = len(t)
x = np.vstack((t, np.ones(N)))
p = np.loadtxt('p.txt')
y = p

def compute_cost_stochastique(x, y, theta):
	hx = np.dot(theta.T, x)
	return np.sum(np.array(hx - y) ** 2) / x.shape[1]

def gradient_descent_stochastique(alpha,x, y, num_iters):
	costs = np.zeros(num_iters)
	i=0
	while i < num_iters:
		for j in range(theta.shape[0]):
			hx = np.dot(theta.T, x)
			diff = y-hx
			theta[j] = theta[j] + (alpha) * np.dot(diff, x[j, :].T)
			i+=1
			costs[i-1] = compute_cost_stochastique(x, y, theta)
			print "iter stocha %s | J_stocha: %.3f" % (i, costs[i-1])
	return theta, costs

if __name__ == '__main__':

	theta = np.zeros((2, 1))
	iterations = 60000
	alpha = 0.0001
	cost = compute_cost_stochastique(x, y, theta)
	print "cout initial gradient stochastique=", cost
	theta,costs = gradient_descent_stochastique(alpha,x, y, iterations)
	print "theta stochastique",theta.T
	# plot regression linaire
	print "Generation d'un graphe de gradient stochastique..."
	plot1 = Generation()
	plot1.plotData(t, p)
	plot1.plotCurveNamed(t.T, np.dot(x.T,theta), "descente gradient stochastique")
	plot1.setLabel("x = temps", "y = position")
	plot1.show()
	#plot iterations et couts stochastique
	pylab.plot(range(iterations), costs)
	pylab.xlabel("#-iterations gradient stocha")
	pylab.ylabel("Couts gradient stocha")
	pylab.show()
	print "Done!"
