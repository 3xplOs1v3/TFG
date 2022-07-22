#!/usr/bin/env python3
{
   "shell_cmd": "gnome-terminal -- bash -c \"python3 -u $file;echo;echo Press Enter to exit...;read\"",
   "file_regex": "^[ ]*File \"(...*?)\", line ([0-9]*)",
   "selector": "source.python",
   
}
from numpy.random import random, normal, choice
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
import math

x0, y0 = -1, -1 # condicion inicial
Dominio = 2
[a,b] = [x0, x0+Dominio] # Dominio de t
h = 0.1 # discretizacion 
c = 5 # numero de coeficientes
p = 20 # numero personas
#N = 100 # numero generaciones

precision = 5


M = []
#M = a+(b-a)*random(p,c) 
#M = [ [ round(a+(b-a)*random(),precision) for i in range(c)] for j in range(p)]
#M = [ [ round(x0+(y0-x0)*random(),precision) for i in range(c)] for j in range(p)]
#M = [ [ round(random(),3) for i in range(c)] for j in range(p)]
#print("matriz randoms",M)

t = Symbol('t')

ysol = [t**i for i in range(c)]
Dy = [y.diff(t) for y in ysol]
vi = [x0**i for i in range(c)]

# discretizacion
values = [{ t : round(a+i*h,precision) } for i in range(int((b-a)/h)+1)]
xk = [round(a+i*h,precision)  for i in range(int((b-a)/h)+1)]


print("tiempos", values)


def min(x):
    mini = 0
    for i in range(len(x)):
        if x[i][1]<x[mini][1]:
            mini = i
    return mini

def suma(x,y): # suma arrays
    res = []
    for i in range(len(x)):
        res.append(round(x[i]+y[i],precision))
    return res


def resta(x,y): # resta arrays
    res = []
    for i in range(len(x)):
        res.append(round(x[i]-y[i],precision))
    return res

def prod(k,x): # k*array
    res = []
    for i in range(len(x)):
        res.append(round(k*x[i],precision))
    return res


def fitness(f):
    ecuacion =  abs(f.diff(t) - 0) 
    pvi = abs(y0-f.subs({t : x0})) 
    return ecuacion+pvi

def evalua(M):
    #evalua
    V = []
    for persona in M:
        val = 0
        candidato = np.dot(ysol,persona)
        for valor in values:
            val += round(abs(fitness(candidato).subs(valor)), precision)
        V.append([persona,round(abs(val),precision)])

    return V

# y = c0 + c1*x + c2*x**2
# y'= c1 + 2*c2*x
def deriva(candidato, xk, n):
    res = 0
    if n == 0:
        for i in range(len(candidato)):
            res+=ci*(xk**i)
    elif n == 1:
        for i in range(len(candidato)):
            res+=i*ci*(xk**(i-1))
    return res

def fit(candidato,xkk):
    ecuacion, derivado, noderivado, pvi, nocero = 0, 0, 0, 0, 0
    for i in range(len(candidato)):
        nocero+=abs(candidato[i])
        if i>0:
            derivado += i*candidato[i]*(xkk**(i-1))
        noderivado += candidato[i]*(xkk**i)
        pvi += candidato[i]*(x0**i)
    if (abs(nocero)<0.5):
        return 999
    return ( 2*abs(derivado-noderivado) +abs(y0-pvi) )


def fitfull(candidato):
    val = 0
    for xkk in xk:
        val += round(fit(candidato, xkk), precision)
    return [candidato, round(abs(val),precision)]


def eval(M):
    V = []
    for persona in M:
        V.append(fitfull(persona))
    return V


def selecciona(V):
    # selecciona
    res = []
    mejores = [] 
    valormejor = []
    for k in range(int(p/2)):
        index = min(V)
        mejores.append(V[index][0])
        valormejor.append(V[index][1])
        res.append([V[index][0],V[index][1]])
        V.pop(index)

    #print(" valor mejores ",mejores)
    #return res
    #print("los duros ",res)
    return mejores

def muta(x, k, step):
    res = x.copy()
    #k = int(random()*len(x)-1)
    for i in range(len(x)):
        #if 0:
        if step < int(N/4):
         #   print("noenetres")
            res[i]=round(normal(res[i],abs( res[i] * (N/(step+1))*(1/10)*max([abs(x0),abs(y0)])  )),precision)
        else:
            res[i]=round(normal(res[i],abs( res[i] * (N/(step+1))*(1/100)*max([abs(x0),abs(y0)])  )),precision)
    #res[k]=round(normal(res[k],abs(res[k]/2)),precision)
    return res

def cruzar(mejores, step):
    N = []
    print("los mejoreeeeees: ", mejores)
    N.append(mejores[0])
    N.append(mejores[1])
    
    for i in range(int((p-2)/2)):
        #N.append(muta(mejores[0],i))
        N.append(muta(mejores[0],choice([i for i in range(c)]), step))
        
        Beta = 0.4
    #for i in range(4):
        N.append(suma(prod(Beta,(resta(N[i],N[i-1]))), N[i-2]))

    
    
    #print("asi keda ",N)
    return N

    

def mezcla(x,y):
    res1, res2 = [], []
    for i in range(len(x)):
        if i<len(x)/2:
            res1.append(x[i])
            res2.append(y[i])
        else:
            res1.append(y[i])
            res2.append(x[i])
    return [res1,res2]


def pinta(ganador):
    t1 = np.arange(x0, x0+Dominio, 0.1)

    plt.figure()
    mia = []
    real = [] # y(t) = k*e^t, k = y0/exp(x0)
    sol = y0/exp(x0)*exp(t)
    for i in range(len(t1)):
        mia.append(ganador.subs({t: t1[i]}))
        real.append(sol.subs({t: t1[i]}))
    plt.plot(t1, mia, 'b', label='mia')
    plt.plot(t1, real, 'k', label='real')
    plt.legend()
    plt.show()

########################################################################################################
crema = []

N = 1000

tnt = 9999
todos = []


for mundos in range(10):
    #M = [ [ round(a+(b-a)*random(),precision) for i in range(c)] for j in range(p)]
    #M = [ [ round(x0+(y0-x0)*random(),precision) for i in range(c)] for j in range(p)]
    #M = [ [ round(np.mean([x0,y0])*random(),precision) for i in range(c)] for j in range(p)]
    M = [ [ round( normal(0, max([abs(x0),abs(y0)])) ,precision) for i in range(c)] for j in range(p)]
    
    print("empiezo connn ",M)
    for veces in range(N):
        print("empiezo con M: ",len(M))
        #V = evalua(M)
        V = eval(M)
        
        #print("M EVALUADOS ",V)
        mejores = selecciona(V)
        
        #print("los mejores: ", mejores)
        M = cruzar(mejores, veces)
        #M = cruze2(mejores)
        
    V = eval(M)
    index = min(V)
    if V[index][1]<tnt : 
        crema.append(V[index][0])
        tnt = V[index][1]
        
    #todos.append(V[index][0])
    #tnt = V[index][1]
    
    

####################################################################################################FIN

print("termineeeeeeeeeeeeeeeeeeeeee")

M = crema.copy()
#M = todos.copy()

V = eval(M)
#print(V)

index = min(V)
mejor = V[index][0]
valormejor = V[index][1]
    

print(" valor mejor ",valormejor)
print(" mejor ", mejor)

ganador=np.dot(ysol, mejor)
print("ganador: ", ganador)

pinta(ganador)






