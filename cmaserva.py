#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 21:23:08 2020

@author: sk
"""
#import numpy as np
from cmaes import CMA 


from numpy.random import random, normal, choice, uniform
#from sympy import lambdify, Symbol


from symengine import *
#from sympy import *

import sys
import math
import numpy as np
import multiprocessing as mp
import pickle

import matplotlib.pyplot as plt

from sympy import re #, evalf, lambdify

err = 9e19

#from symengine import sympify, Lambdify


x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

import copy


# d = 1 -> wi*exp(-ki*(x-ci))
def individuo(genes): # [[genes1],[genes2],...]
    res = []
    global evoluta
    if evoluta==0:
        genes = tri(genes)
        for g in genes:
            res.append(sympify(g[0]*exp(-g[1]*((x-g[2])**2))))
    elif evoluta==1:
        genes = tri(genes)
        expo = 0
        for g in genes:
            res.append(sympify(g[0]*(x**expo)*exp(-g[1]*((x-g[2])**2))))
            expo += 1
    elif evoluta==2:
        for i in range(len(genes)):
            res.append(sympify(genes[i]*(x**i)))
    elif evoluta == 3:
        res.append(genes[0])
        i, m = 1, 1 
        while i<len(genes):
            res.append(genes[i]*cos(m*math.pi*(x-x0[0])/Dominio))
            i+=1
            res.append(genes[i]*sin(m*math.pi*(x-x0[0])/Dominio))
            m+=1
            i+=1
        
    return sum(res)

"""
def individuoTaylor(genes): # [gen0, gen1,..]
    res = []
    for i in range(len(genes)):
        res.append(sympify(genes[i]*(x**i)))
    return sum(res)
"""
     
def caldo(cuantos):
    mundo = []
    global N
    for _ in range(cuantos):
        res = []
        for _ in range(N):
            res.append([round( normal(0, max([abs(x0[0]),abs(y0[0])]) ) ,precision) for _ in range(3)])
        mundo.append(res)
    return mundo

def caldoTaylor(cuantos):
    mundo = []
    global N
    for _ in range(cuantos):
        res = []
        for _ in range(N):
            res.append(round( normal(0, max([abs(x0[0]),abs(y0[0])])/2 ) ,precision) )
        mundo.append(res)
    return mundo



# ecuacion
def fitness(f):
    
    try:
        
        ecuacion = abs(f.diff(x,1)-sin(x*f))
        pvi = 0
        
        for i in range(len(x0)):
            fvi = deriva(f,i)(x0[i]).item(0)
            vi = (y0[i]-fvi)
            if str(vi) == str(1e400*0) or str(vi) == "zoo": #math.isnan(vi)
                return err
            pvi += abs(re(vi))
        
    except:
        return err
        
    return (ecuacion+pvi)

def deriva(f,n):
    return Lambdify(x,f) if n==0 else Lambdify(x,f.diff(x,n))


def cero(f):
    return suma(deriva(f,0)(xk))

def condicion(array):
    res = 0
    for a in array: 
        if a < 0.1: res+=1
    return res

# evalua candidato en todos puntos
def fitea(genes):
    global evoluta
    """
    if evoluta==0: f = sympify(individuo(genes))
    elif evoluta==1: f = sympify(individuoTaylor(genes))
    """
    f = sympify(individuo(genes))
    global xk
    val = 0
    
    escero = 0
    
    f = sympify(f)
    #print("voy con ",f)
    try:
        fu = Lambdify(x,fitness(f))
        val = sum(fu(xk))
        
        # ver si f -> 0 
        #arr = deriva(f,0)(xk[:int(len(xk)/2)])
        arr = deriva(f,0)(xk)
    except:
        print("1 xd")
        return err
    
    
    #val = sum(arr)
    
    if str(val) == str(1e400*0) or val==math.inf: #math.isnan(val)
        print("3")
        return err
    """
    if escero<5/2 * Dominio:
        return err
    """
   # if condicion(arr)/len(arr)  > 0.5:
    #    return err
    return val
 
    

def evalua(mundo):
    pool = mp.Pool(mp.cpu_count())
    try:
        results = pool.map(fitea, [ m for m in mundo ])
    except:
        print("feil")
    pool.close()
    return [[results[i],i] for i in range(len(results))]




def mejores(cuantos, mundo, lista):
    res = []
    for i in range(cuantos):
        res.append(mundo[lista[i][1]])
    return res


def cruza(mejores):
    res = []
    for i in range(int(len(mejores)/2)):
        res.append(m)
    return res
    
def mezclakernels(m1,m2):
    mitad = int(len(m1)/2)
    res1, res2 = [], []
    for i in range(len(m1)):
        if i >= mitad:
            res1.append(m2[i])
            res2.append(m1[i])
        else:
            res1.append(m1[i])
            res2.append(m2[i])
    return [res1,res2]

def mezclacomponentes(m1):
    mitad = int(len(m1)/2)
    for i in range(mitad):
        reserva = m1[i][0]
        m1[i][0] = m1[i+1][0]
        m1[i+1][0] = reserva
        
        
    return m1

def crandom(m1):
    for _ in range(5):
        k1 = int(random()*len(m1))
        k2 = int(random()*len(m1))
        
        c1 = int(random()*3)
        
        reserva = m1[k1][c1]
        m1[k1][c1] = m1[k2][c1]
        m1[k2][c1] = reserva
    return m1

def krandom(m):
    for _ in range(int(len(m)/2)):
        p1 = int(random()*len(m))
        p2 = int(random()*len(m))
        
        k1 = int(random()*len(m[p1]))
        k2 = int(random()*len(m[p2]))
        
        reserva = m[p1][k1]
        m[p1][k1] = m[p2][k2]
        m[p2][k2] = reserva
    
    return m
    
    
        
    

def muta(individuo,prob,var):
    res = []
    for kernel in individuo:
        ker = []
        for gen in kernel:
            if random()<prob:
                gen = round(normal(gen, abs(gen))/var, precision)
            ker.append(gen)
        res.append(ker)
    return res
            
        
        
def norm(tri):
    #tri = sorted(tri)
    res = []
    for t in tri:
        [res.append(ti) for ti in t]
    return res

def tri(norm):
    res = []
    pro = []
    for ni in norm:
        if len(pro)==2: 
            pro.append(ni)
            res.append(pro)
            pro = []
            
        else: pro.append(ni)
    return res
        

        
#x0, y0 = -1,-1 # condicion inicial
#x0, y0 = 0,0 # condicion inicial
#x0, y0 = 0.1, 2.1/sin(0.1) 

x0 = [0]
y0 = [1]
#y0 = [20.1]
#x1, y1 = 0,10
#x0, y0 = 0.1, 20.1

Dominio = 10
[a,b] = [x0[0], x0[0]+Dominio] # Dominio de t
Discretizacion = 99
#h = Dominio/Discretizacion # discretizacion 

precision = 4
xk = sorted(uniform(a, b, Discretizacion))


N = 11 # cuantos kernels

# 0: Gauss. 
# 1: GaussPro ?
# 2: Taylor. 
# 3: Fourier
evoluta = 2

personas = 1000

poblacion = 10

"""
if evoluta == 0 or evoluta == 1: 
    N = int(N/3)
    """


def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2


if __name__ == "__main__":
    if evoluta == 0 or evoluta == 1: 
        optimizer = CMA(mean=np.zeros(3*N), sigma=1.1, population_size=1000)
    else:
        optimizer = CMA(mean=np.zeros(N), sigma=1.3, population_size=1000)

    for generation in range(100):
        solutions = []
        for _ in range(optimizer.population_size):
            xx = optimizer.ask()
            value = fitea(xx)
            solutions.append((xx, value))
            print(f"#{generation} {value} (x1={xx[0]}, x2 = {xx[1]})")
        optimizer.tell(solutions)
print("\n\n\n")
print(individuo(xx))
print(fitea(xx))

f = individuo(xx)

def rmses(f, freal):
    f = Lambdify(x,f)(xk)
    freal = Lambdify(x,freal)(xk)
    
    
    sqr = 0
    for i in range(len(f)):
        sqr+=(f[i]-freal[i])**2
    return sqrt(sqr/len(xk))

def pinta(f, desde, hasta):
    #mundo()
    xx = np.arange(desde, hasta, 0.05)
    plt.plot(xk, Lambdify(x,f)(xk))
    plt.show()

