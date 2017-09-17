from plot import Generation
import numpy as np
import math

def mesureAbs(x,teta,y,N):
	v = y - np.dot(x.T,teta)
	return np.sum(np.absolute(v))/N

def mesureNormal1(x,teta,y,N):
	v = y - np.dot(x.T,teta)
	return math.sqrt(np.sum(np.dot(v.T, v)))/N

def mesureNormal2(x,teta,y,N):
	v = y - np.dot(x.T,teta)
	return np.sum(np.dot(v.T, v))/N

def mesureLinf(x,teta,y):
	v = y - np.dot(x.T,teta)
	return np.amax(np.absolute(v))


# Recuperation donnees

t = np.loadtxt('t.txt')
p = np.loadtxt('p.txt')
a=2
b=3
N=100
x=np.linspace(5,15,N)
y=a*x+b

# Affichage des donnees

print "Generation d'un graphe..."
plot = Generation()
plot.plotData(t, p)
plot.plotCurveNamed(t.T, y.T, "Regression lineaire")
plot.setLabel("x = temps", "y = position")
plot.show()
plot.savefig("MesurePerf.png")
print "Graph fait!"

# Calcul des performances

z = np.ones(len(x))

x1 = np.zeros((2, N))
x1[1,:] = t
x1[0,:] = z

t= np.array([b,a], float)

print "Jlabs =", mesureAbs(x1,t,p,N)
print "Jl1 =", mesureNormal1(x1,t,p,N)
print "Jl2 =", mesureNormal2(x1,t,p,N)
print "Jlinf =", mesureLinf(x1,t,p)
