#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 21:23:08 2020

@author: sk
"""

from numpy.random import random, normal, choice, uniform
#from sympy import lambdify, Symbol


from symengine import *
#from sympy import *




import sys
import math
import numpy as np
import multiprocessing as mp
import pickle
#import spacy

#from matplotlib import *
#from pylab import *
#from numpy import *

#from uncertainties import ufloat


import cmath


from sympy import re #, evalf, lambdify



#from symengine import sympify, Lambdify

x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

#E = ["EOE","(E)","F(E)","D","x","y","z"] # mod 7
E = ["EOE","(E)","F(E)","D","V","-E"] # mod 5
#E = ["EOE","(E)","F(E)","D","x"] # mod 5
##     0     1      2    3   4   5   6
#O = ["+","-","*","/"] # mod 4
O = ["+","-","*","/","**"] # mod 4

##    0   1   2   3
F = ["sin","cos","exp","log","sqrt"] # mod 4
##     0     1     2     3
D = ["0","1","2","3","4","5","6","7","8","9"] # mod 10

#V = ["x","y","z"]
V = ["x","-x"]



dict = {}
dict["E"] = E
dict["O"] = O
dict["F"] = F
dict["D"] = D
dict["V"] = V


error = -999
err = 9e15
#hay mayuscula 
def mayus(s): 
    s = str(s)
    res = 0
    for si in s:
        if si.isupper():
            res = 1
            break
    return res

#dado parser devuelve expresion
def decode(cromosoma):
    string = ""
    estado = E
    dale = 0
    i = 0
    cromos = []
    letra = "E"
    for c in cromosoma:
        if i!=0 and not mayus(string):
            break
        else:
            cromos.append([letra, c%len(estado)])
        
        if dale:
            string= string[:i] + estado[c%len(estado)] + string[i+1:]
            i=0
        else:
            string+=(estado[c%len(estado)])
        for s in string:
            if s.isupper():
                letra = s
                estado = dict[s]
                dale = 1
                break
            else:
                i+=1
                dale = 0
    return [string,cromos[:]]
    #return [sympify(string),cromos[:]]
    

# array de randoms
def c0(N):
    ci = []
    ci.append(int(random()*8))
    for k in range(N):
        ci.append(int(20*random()))
    return ci

def basicas():
    ci = []
    ci.append(2)
    ci.append(int(random()*len(F)))
    ci.append(4)
    ci.append(int(random()*len(V)))
    return ci

def inversas():
    ci = []
    ci.append(0)
    ci.append(3)
    ci.append(1)
    ci.append(3)
    ci=ci+basicas()
    return ci

def nobasicas():
    ci = []
    ci.append(0)
    ci=ci+basicas()
    ci.append(int(random()*len(O)))
    ci=ci+basicas()
    return ci

def fof():
    ci = []
    ci.append(2)
    ci.append(int(random()*len(F)))
    ci=ci+basicas()
    return ci


def caldo():
    res = []
    basicas = []
    # polinomios (grado 3)
    [basicas.append([0,4,v,4,3,d]) for v in range(len(V)) for d in range(4)]
    #basicas (x y -x)
    [basicas.append( [2,k,4,v] ) for k in range(len(F)) for v in range(len(V)) ]
    #-basicas
    [basicas.append( [5,2,k,4,v] ) for k in range(len(F)) for v in range(len(V)) ]
    #basica O basica
    bob = []
    #[bob.append([0]+b1+[o]+b2) for b1 in basicas for o in range(len(O)) for b2 in basicas]
    
    fof = []
    #[fof.append([2,f]+b) for f in range(len(F)) for b in basicas]
    #[fof.append([2,f]+b) for f in range(len(F)) for b in bob]
    res = basicas+bob+fof
    #return [basicas,bob,fof]
    return [decode(i) for i in res]

# array valido
def valido(array):    

    global N
    kk = decode(array)
    try:
        strin = sympify(kk[0])
    except:
        strin = error
    if mayus(strin) == 1 or "zoo" in str(strin) or "nan" in str(strin):
        strin = error
    return strin


# crea nuevo individuo
def fresh():
    global M, N
    kk = decode(c0(N))
    try:
        strin = sympify(kk[0])
    except:
        strin = error
    while (strin == error or strin in M or mayus(strin) == 1 or "zoo" in str(strin) or "nan" in str(strin)):
        #print("no vale")
        kk = decode(c0(N))
        try:
            strin = sympify(kk[0])
        except:
            strin = error
            continue
    M.append(strin)     
    return [strin,kk[1]]


# ecuacion
def fitness(f):
    f = sympify(f)
    #ecuacion =  abs(f.diff(x) - x**2*f ) 
    #ecuacion =  abs(f.diff(x) - (2*x-f)/x ) 
    #ecuacion =  abs(f.diff(x) - f ) 
    """
    if derivable(f):
        ecuacion = abs(f.diff(x) - (-f/5 + exp(-x/5)*cos(x)))
    else:
        from sympy import Symbol
        ecuacion = abs(f.diff(x) - (-f/5 + exp(-x/5)*cos(x)))
        from symengine import Symbol
        """
    
    #ecuacion = abs(f.diff(x) - (-f/5 + exp(-x/5)*cos(x)))
    #ecuacion = abs(f.diff(x)-(f**2))
    #ecuacion = abs(f.diff(x)-(1-f*cos(x))/sin(x))
    #ecuacion = abs(f.diff(x) - (-f/5 +exp(-x/5)*cos(x)))
    #ecuacion = abs(f.diff(x) - exp(-x**2))
    #ecuacion = abs(f.diff(x)**2+log(f)-cos(x)**2-2*cos(x)-1-log(x+sin(x)))
    #ecuacion = abs(f.diff(x).diff(x)+100*f)
    #x'' = x - x^3 + x' (4 - 5 x^2)/20; x(0)=1.5, x(0)'=0;
    
    
    
    try:
        #ecuacion = (f.diff(x) - (-f/5 + exp(-x/5)*cos(x)))**2
        ecuacion = abs(f.diff(x,2)+100*f)
        #ecuacion = abs(f.diff(x,2) -f + f**3 - f.diff(x)*(4 - 5*f**2)/20 )
        #fvi = f.subs({x : x0})
        fvi = Lambdify(x,f)(x0[0]).item(0)
        vi = (y0[0]-fvi)
        if str(vi) == str(1e400*0) or str(vi) == "zoo": #math.isnan(vi)
            return err
        pvi = abs(re(vi))
        
        for i in range(1,len(x0)):
            fvi = Lambdify(x,f.diff(x,i))(x0[i]).item(0)
            vi = (y0[i]-fvi)
            if str(vi) == str(1e400*0) or str(vi) == "zoo": #math.isnan(vi)
                return err
            pvi += abs(re(vi))
        
    except:
        return err
        
    
    
    #pvi += abs(y1-f.diff(x).subs({x : x1}))
    
    return (10*ecuacion+pvi)

def deriva(f,n):
    return 


# evalua candidato en todos puntos
def fit(f):
    
    global xk
    val = 0
    escero = 0
    #err = 9999999999
    
    f = sympify(f)
    
    try:
        fu = lambdify(x,fitness(f))
    except:
        #print("1")
        return err
    for xkk in xk:
        try:
            val+=abs(round(fu(xkk),precision))
            #escero+=abs(round(f.diff(x).subs({x:xkk})))
        except:
            #print("2")
            return err
    if math.isnan(val):
        #print("3")
        return err
    #if escero<0.5 and val!=0:
    #    return err
    return round(val,precision)

def fit1(f):
    
    global xk
    val = 0
    escero = 0
    #err = 9999999999
    
    f = sympify(f)
    
    #try:
    fu = Lambdify(x,fitness(f))
    #except:
     #   print("1")
      #  return err
    for xkk in xk:
        try:
            val+=abs(round(fu(xkk).item(0),precision))
            #escero+=abs(round(f.diff(x).subs({x:xkk})))
        except:
            print("2")
            return err
    if math.isnan(val):
        print("3")
        return err
    #if escero<0.5 and val!=0:
    #    return err
    return round(val,precision)

def fit2(f):
    
    global xk
    val = 0
    escero = 0
    #err = 9999999999
    
    f = sympify(f)
    
    #try:
    #fu = Lambdify(x,fitness(f))
    #ex = fitness(f)
    #except:
     #   print("1")
      #  return err
    ex = fitness(f)
    for xkk in xk:
        try:
            #print(ex)
            #print(round(ex.subs({x:xkk}),precision))
            val+=abs(ex.subs({x:xkk}).n(5, real=True))
            #print("asi va ",val)
            #escero+=abs(round(f.diff(x).subs({x:xkk})))
        except:
            print("2")
            return err
    if math.isnan(val):
        print("3")
        return err
    #if escero<0.5 and val!=0:
    #    return err
    return val

# evalua candidato en todos puntos
def fitea(f):
    
    global xk
    val = 0
    escero = 0
    #err = 9999999999
    
    f = sympify(f)
    #print("voy con ",f)
    try:
        fu = Lambdify(x,fitness(f))
    except:
        #print("1 xd")
        return err
    
    try:
        val = sum(fu(xk))
    except:
        #print("2")
        return err
    
    #val = sum(arr)
    
    if str(val) == str(1e400*0): #math.isnan(val)
        #print("3")
        return err
    #if escero<0.5 and val!=0:
    #    return err
    return val
    

# evalua candidato en todos puntos
def fitop(f):
    
    global xk
    val = 0
    escero = 0
    #err = 9999999999
    
    #f = sympify(f)
    
    try:
        fu = lambdify(x,fitness(f))
    except:
        return err
    for xkk in xk:
        try:
            val+=abs(round(fu.subs({x:xkk}),precision))
            #escero+=abs(round(f.diff(x).subs({x:xkk})))
        except:
            return err
    if math.isnan(val):
        return err
    #if escero<0.5 and val!=0:
    #    return err
    return round(val,precision)

def fitt(f):
    
    global xk
    val = 0
    escero = 0
    #err = 9999999999
    
    f = sympify(f)
    ecuacion = abs(f.diff(x) - (-f/5 +exp(-x/5)*cos(x)))
    pvi = abs(y0-f.subs({x : x0})) 
    
    try:
        fu = lambdify(x,ecuacion)
    except:
        return err
    for xkk in xk:
        try:
            val+=abs(round(fu(xkk),precision))
            #escero+=abs(round(f.diff(x).subs({x:xkk})))
        except:
            return err
    if math.isnan(val):
        return err
    #if escero<0.5 and val!=0:
    #    return err
    return round(val,precision)

def klkfitop(f):
    
    f = sympify(f)
    
    ecuacion = (f.diff(x) - (-f/5 + exp(-x/5)*cos(x)))**2    
    pvi = abs(y0-f.subs({x : x0})) 
    #pvi += abs(y1-f.diff(x).subs({x : x1})) 
    
    global xk
    val = 0
    escero = 0
    
    try:
        fu = lambdify(x,ecuacion)
    except:
        return err
    for xkk in xk:
        try:
            val+=abs(round(fu(xkk),precision))
        except:
            return err
    if math.isnan(val):
        return err
    if val<1 and pvi!=0:
        return err
    return [round(val,precision),round(pvi,precision)]

#evalua lista de candidatos
def evalua(M):
    V = []
    i=0
    for persona in M:
        #print(i)
        #i+=1
        #print("voy con ", persona[0])
        V.append([persona, round(fitea(persona[0]),precision)])
    return V

def evalua1(M):
    V = []
    i=0
    for persona in M:
        V.append([persona, round(fit(persona[0]),precision)])
        
    return V

# index del mejor candidato
def mijor(x):
    mini = 0
    for i in range(len(x)):
        if math.isnan(x[i][1]):
            continue
        if x[i][1]<x[mini][1]:
            mini = i
    return mini


# selecciona de los evaluados tantos
def selecciona(V, tantos):
    uve = V[:]
    res = []
    mejorees = [] 
    finn = []
    otro = []
    #valormejor = []
    
    listado = []
    k = 0
    while k < tantos:
        index = mijor(uve)
        #print(" el puto mejor es ", uve[index][0][0], " = ", sympify(decode(genes(uve[index][0])))[0])
        vale = uve[index][0][:]
        #print(" ecawonnndns es ", vale[0], " = ", sympify(decode(genes(vale)))[0])
        mejorees.append(vale)
        #mejorees.append(uve[index][0])
        #valormejor.append(V[index][1])
        #res.append([V[index][0],V[index][1]])
        finn.append(vale.copy())
        vaa = uve.pop(index)
        if sympify(vaa[0][0]) not in listado:
            otro.append(vaa[0])
            listado.append(sympify(vaa[0][0]))
            k+=1

    #print(" valor mejores ",mejores)
    #return res
    #print("los duros ",res)
    #print(" selecciona COMPRUEBO A LA SALIDA ")
    #for mi in finn:
    #    print(mi[0], " = " , decode(genes(mi))[0])
    #return mejorees
    return otro

#muta
def muta(persona, caso):
    
    personaa = persona.copy()
    cromosomas = personaa[1]
    sec = []
    
    #print("muto ", personaa[0], " con caso ",caso)
    #print("osea ", decode(genes(personaa))[0])
    
        
    if caso==0:
        if random()<0.5: # D O persona
            sec=[]
            trozo = []
            trozo.append(0) # E O E
            trozo.append(3) # E -> D
            trozo.append(int(random()*len(D))) # D
            trozo.append(int(random()*len(O))) # O
            trozo.append(1) # ( E )
            for c in cromosomas:
                trozo.append(c[1])
            sec = trozo
        else: # persona O D 
            sec=[]
            trozo = []
            trozo.append(0) # E O E
            trozo.append(1) # E -> ( E )
            for c in cromosomas:
                trozo.append(c[1])
            trozo.append(int(random()*len(O))) # O
            trozo.append(3) # E -> D
            trozo.append(int(random()*len(D))) # D
            sec = trozo
        
    
    # # MISMA D O V
    # elif caso==2:
    #     sec=[]
    #     serva  = []
    #     digit, op = error, error
    #     for c in cromosomas:
    #         serva.append(c[1])
    #         var = c[1]
    #         if c[0]=='V': # para todas las V que haya ?
    #             serva.pop()
    #             serva.pop()
    #             serva.append(1) # E -> ( E )
    #             serva.append(0) # E -> E O E
    #             serva.append(3) # E -> D
    #             if digit == error and op == error:
    #                 digit = int(random()*len(D))
    #                 op = int(random()*len(O))
    #             serva.append(digit) # D 
    #             serva.append(op) # O
    #             serva.append(4) # E -> V
    #             serva.append(var) # misma var
    #     sec = serva
        
        
    # # MISMA V O D
    # elif caso==3:
    #     sec=[]
    #     serva  = []
    #     digit, op = error, error
    #     for c in cromosomas:
    #         serva.append(c[1])
    #         var = c[1]
    #         if c[0]=='V': # para todas las V que haya ?
    #             serva.pop()
    #             serva.pop()
    #             serva.append(1) # E -> ( E )
    #             serva.append(0) # E -> E O E
    #             serva.append(4) # E -> V
    #             serva.append(var) # misma var
    #             if digit == error and op == error:
    #                 digit = int(random()*len(D))
    #                 op = int(random()*len(O))
    #             serva.append(op) # O
    #             serva.append(3) # E -> D
    #             serva.append(digit) # D 
    #     sec = serva
        
    # randomiza D u O
    # elif caso == 4:
    #     serva = []
    #     sec=[]
    #     for c in cromosomas:
    #         if c[0]=='O':
    #             n=int(len(O)*random())
    #         elif c[0]=='D':
    #             n=int(len(D)*random())
    #         else:
    #             n=c[1]
    #         serva.append(n)
    #     sec = serva

    
    # D O V kiza
    elif caso==1:
        if random()<0.5:
            sec=[]
            serva  = []
            digit, op = error, error
            for c in cromosomas:
                serva.append(c[1])
                var = c[1]
                if c[0]=='V' and random()>0.5: 
                    serva.pop()
                    serva.pop()
                    serva.append(1) # E -> ( E )
                    serva.append(0) # E -> E O E
                    serva.append(3) # E -> D
                    if digit == error and op == error:
                        digit = int(random()*len(D))
                        op = int(random()*len(O))
                    serva.append(digit) # D 
                    serva.append(op) # O
                    serva.append(4) # E -> V
                    serva.append(var) # misma var
            sec = serva
        else: # V O D kiza
            sec=[]
            serva  = []
            digit, op = error, error
            for c in cromosomas:
                serva.append(c[1])
                var = c[1]
                if c[0]=='V' and random()>0.5: 
                    serva.pop()
                    serva.pop()
                    serva.append(1) # E -> ( E )
                    serva.append(0) # E -> E O E
                    serva.append(4) # E -> V
                    serva.append(var) # misma var
                    if digit == error and op == error:
                        digit = int(random()*len(D))
                        op = int(random()*len(O))
                    serva.append(op) # O
                    serva.append(3) # E -> D
                    serva.append(digit) # D 
            sec = serva
        
    # randomiza
    elif caso == 2:
        serva = []
        sec=[]
        for c in cromosomas:
            #if c[0]=='F':
             #   if random()<0.5: n=int(len(F)*random())
            #elif c[0]=='O':
             #   if random()<0.5: n=int(len(O)*random())
            if c[0]=='D':
                n=int(len(D)*random())
            else:
                n=c[1]
            serva.append(n)
        sec = serva
    
    
    # persona O E
    elif caso==3:
        sec=[]
        sec.insert(0,0) 
        for c in cromosomas:
            sec.append(c[1])
        posible = c0(N)
        while valido(posible) == error:
            posible = c0(N)
        kk = decode(posible)
        
        trozo = []
        for k in kk[1]:
            trozo.append(k[1])
        sec.append(int(random()*len(O)))
        sec = sec+trozo
    
    #print("y queda ", decode(sec)[0])
    #sec = []
    return decode(sec)


# cruza
def cruza(mejores):
    #print("COMPRUEBO MEJORES")
    #for m in mejores:
    #    print(m[0], " = ", sympify(decode(genes(m))[0]))
    expr = []
    mejorees = mejores[:]
    global M
    res = []
    for m in mejorees:
        #res.append(sympify(m))
        #res.append(sympify(muta(m)))
        #print("voy a llamar a mutar con ", m)
        #print(" osea ", decode(genes(m)))
        res.append(m) # no lo toco
        expr.append(m[0])
        nuevo = muta(m,int(random()*6 % 4))
        try:
            new = sympify(nuevo[0])
        except:
            new = error
        for k in range(len(mejores)-2):
            while (new == error or new in expr): #new in M
                #print("nito fresh")
                #print("CONCATENA ")
                #print("stuck, ",new," ta en ", M)
                #time.sleep(2)
                nuevo = muta(m,int(random()*6 % 4))
                #nuevo = muta(m,k)
                try:
                    new = sympify(nuevo[0])
                except:
                    new = error
                    continue
            #M.append(new)
            expr.append(new)
            res.append([new, nuevo[1]])
            
        
        #fresh = nuevo()
        #M.append(fresh[0])
            
        #res.append(fresh)
        #res.append(fresh())  
        #res.append(valido(c0(N)))
        res.append(decode(fof()))
        #res.append(muta(m))
    return res


def funciones(mundo):
    for m in mundo:
        print(sympify(m[0]))
        
        
# paralelo
def evaluaop(mundo):
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(fit, [m[0] for m in mundo])
    V = []
    for i in range(len(mundo)):
        V.append([mundo[i],results[i]])
    pool.close()
    return V

def eevaluaop(mundo):
    pool = mp.Pool(mp.cpu_count())
    try:
        results = pool.map(fitea, [ m[0] for m in mundo ])
    except:
        print("feil")
    V = []
    for i in range(len(mundo)):
        #print(i)
        V.append([mundo[i],results[i]])
    pool.close()
    return V

# paralelo
def evaluaoop(mundo):
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(fitop, [m[0] for m in mundo])
    pool.close()
    V = []
    for i in range(len(mundo)):
        V.append([mundo[i],results[i]])
    return V

def evaluaopp(mundo):
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(fit1, [m[0] for m in mundo])
    
    V = []
    for i in range(len(mundo)):
        V.append([mundo[i],results[i]])
    pool.close()
    return V


def evv(mundo):
    res = []
    pool = mp.Pool(mp.cpu_count())
    [res.append( [ m, pool.map(fitop, [m[0] for m in mundo ]) ] ) for m in mundo]
    pool.close()
    return res

def crea(N):
    mundo = []
    
    for i in range(10):
        mundo.append(decode(basicas()))
    for i in range(200):
        mundo.append(decode(nobasicas()))
    for i in range(50):
        mundo.append(decode(fof()))
    for i in range(10):
        mundo.append(decode(inversas()))
    for i in range(N):
        mundo.append(fresh())
    return mundo


def genes(persona):
    res = []
    for estado in persona[1]:
        res.append(estado[1])
    return res

def comrpuebaV(V):
    for v in V:
        print(v[0][0]," = ",sympify(decode(genes(v[0]))[0]))
        print(v[0][0]==sympify(decode(genes(v[0]))[0]))

def comrpuebaS(S):
    for s in S:
        print(s[0]," = ",sympify(decode(genes(s))[0]))
        print(s[0]==sympify(decode(genes(s))[0]))
        
def criva(gentes):
    res = []
    for g in gentes:
        kk = g.copy()
        try:
            strin = sympify(g[0])
        except:
            strin = error
    
        # or strin in M
        while (strin == error  or mayus(strin) == 1 or "zoo" in str(strin) or "nan" in str(strin)):
            kk = decode(c0(N))
            try:
                strin = sympify(kk[0])
            except:
                strin = error
                continue
        res.append(kk)
    return res

def limpia(ppl):
     return [decode(genes(p)) for p in ppl]

        
N = 30 # numero cromosomas
#x0, y0 = -1,-1 # condicion inicial
#x0, y0 = 0,0 # condicion inicial
#x0, y0 = 0.1, 2.1/sin(0.1) 

x0 = [0,0]
y0 = [0,10]
#x1, y1 = 0,10
#x0, y0 = 0.1, 20.1

Dominio = 5
[a,b] = [x0[0], x0[0]+Dominio] # Dominio de t
Discretizacion = 499
#h = Dominio/Discretizacion # discretizacion 

precision = 4
xk = uniform(a, b, Discretizacion)


M = [] # M de funciones

personas = 1000
    
Va = []

#mejores = []



####################################################################
####################################################################
#mundo = []
#mundo = crea(personas)
mundo = caldo()
#mundo = [decode(i) for i in xd]

#N = 20


from time import sleep

for i in range(100):
    
    print("------------------  GENERACIONNN: ",i," ------------------------------------")

    Va = eevaluaop(mundo)

    mejores = selecciona(Va,10)
    print("LOS MEJORES ",funciones(mejores))
    #print("")
    #print("")
    #print("ganador: ", mejores[0][0], " fit: ", fit(mejores[0][0]))

    mun = cruza(mejores)
    mundo = criva(mun)
    mundo = limpia(mundo)

    #print("cruzados ", funciones(mundo))
    
    if fitea(mejores[0][0])<1e-3:
        print("yata no sigas, aki ta ", mejores[0])
        break
    #sleep(5)

ganador = selecciona(evaluaop(mundo),1)
print("el ganador: ", sympify(ganador[0][0]))
print(" con val ", fitea(ganador[0][0]))


import os
import sys

def derivable(f):
    #f = sympify(sin(x))
    #f = sympify((-x)**(-1/2))
    r, w = os.pipe() 
      
    #Creating child process using fork 
    processid = os.fork() 
    
    if processid: 
        os.close(w) 
        r = os.fdopen(r) 
        stri = r.read() 
        if stri == "":
            #print("ha palmao")
            return False
        else:
            return True
    else: 
        os.close(r) 
        w = os.fdopen(w, 'w') 
        print ("") 
        msg = f.diff(x)
        w.write("1") 
        w.close()
        #print("todo ok")
        #sys.exit(0)
        return
        #exit()
        #quit()
    return True

def deriva(f):
    try:
        f.diff(x)
        return True
    except:
        return False