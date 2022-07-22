from numpy.random import random, normal, choice, uniform
import sys
import math
import numpy as np
import multiprocessing as mp
import pickle
from symengine import *
from sympy import re #, evalf, lambdify


x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

E = ["EOE","(E)","F(E)","D","V","-E"] # mod 6
##     0     1     2     3   4    5
O = ["+","-","*","/","**"] # mod 5
##    0   1   2   3    4
F = ["sin","cos","exp","log","sqrt"] # mod 5
##     0     1     2     3     4
D = ["0","1","2","3","4","5","6","7","8","9"] # mod 10
#D = ["0","1","2","3","4","5","6","7","8","9","(-1)","(-2)","(-3)","(-4)","(-5)","(-6)","(-7)","(-8)","(-9)"] 

V = ["x","-x"]
##    0    1


dict = {}
dict["E"] = E
dict["O"] = O
dict["F"] = F
dict["D"] = D
dict["V"] = V


# devuelve el index de la primera mayuscula
# o -1 si no hay mayusculas    
def mayus(s):
    s = str(s)
    for i in range(len(s)):
        if s[i].isupper(): return i
    return -1

# decodifica un individuo en el Automata Finito
# Genotipo -> Fenotipo
def decode(cromos):
    def decoder(cromos, res, arbol): # recursivo
        if cromos == [] or mayus(res)==-1: return [res,arbol]
        else: 
            i = mayus(res) # index primera letra mayuscula
            estado = res[i] # estado actual del automata
            arbol.append([estado, cromos[0]%len(dict[estado])])
            return decoder(cromos[1:], res[:i]+dict[estado][cromos[0]%len(dict[estado])]+res[i+1:], arbol)
    return decoder(cromos[1:], dict["E"][cromos[0]%len(dict["E"])], [['E',cromos[0]%len(dict["E"])]])


def genes(persona):
    res = []
    for estado in persona[1]:
        res.append(estado[1])
    return res
    
#0 basicas, 1 bob, 2 fof, else todos
def caldo(caso):
    res = []
    basicas = []
    # polinomios (grado 3)
    [basicas.append([0,4,v,4,3,d]) for v in range(len(V)) for d in range(4)]
    #basicas (x y -x)
    [basicas.append( [2,k,4,v] ) for k in range(len(F)) for v in range(len(V)) ]
    #-basicas
    [basicas.append( [5,2,k,4,v] ) for k in range(len(F)) for v in range(len(V)) ]
    #basicas inversas
    inversas = []
    [inversas.append([0,3,1,3]+b) for b in basicas]
    #basica O basica
    bob = []
    [bob.append([0]+b1+[o]+b2) for b1 in basicas for o in range(len(O)) for b2 in basicas]
    
    fof = []
    [fof.append([2,f]+b) for f in range(len(F)) for b in basicas]
    [fof.append([2,f]+b) for f in range(len(F)) for b in bob]
    
    basicas+=inversas
    bob+=inversas
    
    if caso == 0: res = basicas
    elif caso == 1: res = bob
    elif caso == 2: res = fof
    else: return [[decode(i) for i in basicas]]+[[decode(i) for i in bob]]+[[decode(i) for i in fof]]
    return [decode(i) for i in res]

def basicas():
    ci = []
    ci.append(2)
    ci.append(int(random()*len(F)))
    ci.append(4)
    ci.append(int(random()*len(V)))
    return ci

def fof():
    ci = []
    ci.append(2)
    ci.append(int(random()*len(F)))
    ci=ci+basicas()
    return ci

def deriva(f,n):
    return Lambdify(x,f) if n==0 else Lambdify(x,f.diff(x,n))

x0 = [0]
y0 = [1]
# ecuacion
def fitness(f):
    alfa, beta = 1, 1
    try:
        ecuacion = abs(f.diff(x,) -sin(x*f) )
        #ecuacion = abs(f.diff(x,2) -2*f.diff(x,1)+2*f-exp(2*x)*sin(x))
        #ecuacion = abs(f.diff(x) - (-f/5 + exp(-x/5)*cos(x)))
        
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


# evalua candidato en todos puntos
def fitea(f):
    
    global xk
    val = oo
    
    escero = 0
    
    f = sympify(f)
    try:
        fu = Lambdify(x,fitness(f))
        val = sum(fu(xk))
        
        
        #anti tramposos
        arr = deriva(f,0)(xk)
        #arr = arr[:int(len(arr)/2)]
    except:
        return err
    
    
    if str(val) == str(1e400*0) or val==math.inf: #math.isnan(val)
        return err
    # anti tramposos ke % debajo de 0.5
    if condicion(arr, 0.5)/len(arr)  > 0.5:
        return err
    return val

#para tramposos
def condicion(array, menor):
    res = 0
    for a in array: 
        if a < menor: res+=1
    return res

def evalua(mundo):
    pool = mp.Pool(mp.cpu_count())
    try:
        results = pool.starmap(fitea, [ [m[0]] for m in mundo])
        #results = pool.map(fitea, [ m[0] for m in mundo ])
    except:
        print("feil")
    pool.close()
    return [[results[i],i] for i in range(len(results))]

def best(xd,l,cuantos):
    res = []
    listado = []
    ctos,i = 0,0
    while ctos < cuantos:
        nuev = sympify(xd[l[i][1]][0])
        if nuev not in listado:
            listado.append(nuev)
            res.append(xd[l[i][1]])
            ctos+=1
        i+=1
    return res

def muta(persona, caso):
    
    personaa = persona.copy()
    cromosomas = personaa[1]
    sec = []
    
    #print("muto ", personaa[0], " con caso ",caso)
    #print("osea ", decode(genes(personaa))[0])
    
    if caso==0:
        if random()<0.5: # D O persona
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
            trozo = []
            trozo.append(0) # E O E
            trozo.append(1) # E -> ( E )
            for c in cromosomas:
                trozo.append(c[1])
            trozo.append(int(random()*len(O))) # O
            trozo.append(3) # E -> D
            trozo.append(int(random()*len(D))) # D
            sec = trozo
        
    # D O V kiza
    elif caso==1:
        if random()<0.5:
            serva  = []
            digit, op = error, error
            for c in cromosomas:
                serva.append(c[1])
                var = c[1]
                if c[0]=='V' and (op==error or random()>0.3): 
                    serva.pop()
                    serva.pop()
                    serva.append(1) # E -> ( E )
                    serva.append(0) # E -> E O E
                    serva.append(3) # E -> D
                    if digit == error and op == error: # mismo D y O
                        digit = int(random()*len(D))
                        op = int(random()*len(O))
                    serva.append(digit) # D 
                    serva.append(op) # O
                    serva.append(4) # E -> V
                    serva.append(var) # misma var
            sec = serva
        else: # V O D kiza
            serva  = []
            digit, op = error, error
            for c in cromosomas:
                serva.append(c[1])
                var = c[1]
                if c[0]=='V' and random()>0.3:
                #if c[0]=='V' and (op==error or random()>0.3): 
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
        for c in cromosomas:
            if c[0]=='F':
                if random()<0.5: n=int(len(F)*random())
            elif c[0]=='O':
                if random()<0.5: n=int(len(O)*random())
            elif c[0]=='D':
                n=int(len(D)*random())
            else:
                n=c[1]
            serva.append(n)
        sec = serva
    
    
    # persona O E
    elif caso==3:
        if random()<0.5: 
            sec.append(0)
            #sec.insert(0,0) 
            for c in cromosomas:
                sec.append(c[1])
            sec.append(int(random()*len(O)))
            sec = sec+basicas()
        else:
            sec.append(0)
            sec+=basicas()
            sec.append(int(random()*len(O)))
            for c in cromosomas:
                sec.append(c[1])
                
    res = decode(sec)
    
    try: return res
    except: return err

def cruzaop(mejores, pesos):
    res = []
    mejorees = mejores[:]
    results = []
    Mutados = 80
    
    pool = mp.Pool(mp.cpu_count())
    try:
        results = pool.starmap(muta, [ [m, choice([0,1,2,3],  p = pesos) ] for m in mejorees for _ in range(Mutados)])
    except:
        print("feil")
    
    #print("aki van ", len(results))
    pool.close()
    
    hh = 0
    
    #[print("aki tienes results ", resul[0]) for resul in results]
    for i in range(len(mejorees)):
        
        res.append(mejorees[i])
        
        for k in range(Mutados):
            nuevo = results[hh]
            hh+=1
            #new = vale(nuevo)
            try:
                new = sympify(nuevo[0])
            except:
                new = error
            while new == error:
                #print("entro while ", nuevo[0])
                nuevo = muta(mejorees[i],int(random()*6 % 4))
                #new = vale(nuevo)
                try:
                    new = sympify(nuevo[0])
                except:
                    new = error
            
            res.append(nuevo)
    
    
            
        res.append(decode(fof()))

    return res

def criva(gentes):
    wtf = 0
    global caldero
    res = []
    for g in gentes:
        kk = g.copy()
        
        try:
            strin = sympify(g[0])
        except:
            strin = error
    
        # or strin in M
        #string = str(strin)
        while (strin == error  or mayus(strin) == 1 or "zoo" in str(strin) or "nan" in str(strin)):
            wtf+=1
            #print("MUCHO O KE")
            #kk = decode(c0(N))
            #kk = caldero[int(random()*len(caldero))]
            kk = decode(basicas())
            try:
                strin = sympify(kk[0])
            except:
                strin = error
                continue
            
        
        #kk = decode(basicas())
        #kk = caldero[int(random()*len(caldero))]
        res.append(kk)
    #print(wtf,"!!!!!")
    return [decode(genes(p)) for p in res]


error = -999
err = 9e15

N = 30 # numero cromosomas
#x0, y0 = -1,-1 # condicion inicial
#x0, y0 = 0,0 # condicion inicial
#x0, y0 = 0.1, 2.1/sin(0.1) 


#y0 = [20.1]
#x1, y1 = 0,10
#x0, y0 = 0.1, 20.1

Dominio = 5
[a,b] = [x0[0], x0[0]+Dominio] # Dominio de t
Discretizacion = 99
#h = Dominio/Discretizacion # discretizacion 

precision = 4
xk = sorted(uniform(a, b, Discretizacion))


M = [] # M de funciones

personas = 1000
    
Va = []

#mejores = []


####################################################################
####################################################################

semillas = caldo(-1)
#basicass = semillas[0]

def mundo1(I, generaciones):
    mundo = semillas[I]
    #reserva = selecciona(eevaluaop(caldo(1)),20)
    reserva = best(semillas[1],sorted(evalua(semillas[1])),20)
    #reserva1 = selecciona(eevaluaop(caldo(-1)),20)
    reserva+=semillas[0]
    
    
    
    for i in range(generaciones):
        
        def funciones(mundo):
            [print(m[0]) for m in mundo]
                
        
        print("------------------  GENERACIONNN: ",i," ------------------------------------")
        #print(len(mundo))
        """
        Va = eevaluaop(mundo)
    
        print("fin evalua, cogo mejores")
        mej = selecciona(Va,100)+reserva
        print("LOS MEJORES ",funciones(mej))
        mun = cruza(mej)
        """
        Va = evalua(mundo)
        #Va = evaluajuko(mundo)
        print("fin evalua, cogo mejores")
        
        mej = best(mundo,sorted(Va),10)+reserva
        print(funciones(mej[:10]))
        
        print("voy a cruzar")
        pesos = [0.1,0.3,0.3,0.3]
        mun = cruzaop(mej, pesos)
        
        #print("")
        #print("")
        #print("ganador: ", mejores[0][0], " fit: ", fit(mejores[0][0]))
    
        print("voy a crivar")
        
        mundo = criva(mun)
        #mundo = limpia(mundo)
    
        #print("cruzados ", funciones(mundo))
        
        if fitea(mej[0][0])<1e-3:
            print("ya está no sigas: ", mej[0])
            break
        #sleep(5)
        
import matplotlib.pyplot as plt
def pinta(f, desde, hasta):
    #mundo()
    xx = np.arange(desde, hasta, 0.005)
    plt.plot(xx, Lambdify(x,f)(xx))
    plt.show()

sol1 = sqrt(6)/sqrt(x**2 -1 +7*exp(-x**2))
sol2 = (sqrt((sqrt(7 + (2/7)*x**2))**sin(x)))**((sqrt(x**2))**((exp((-2/7)*x**2))**sin(x)))
sol3 = (sqrt((3*sqrt(x))**sin(x)))**(sin(x)**(sin(x)**(sqrt(x))))
sol4 = 2**(sin(x)*sin((3/4)*x))
sol5 = (exp(x))**(-exp(-x**3)) + sin(x)
sol6 = sin(x)*sin((1/3)*x)**((exp((-1/3)*x))**(exp(x))) + (1/60466176)*x**5 + cos((1/42)*x)
sol6 =  sin(x)**2*sin((1/5)*x)**(exp(-x)) + exp((-1/36)*x) + sin((1/8)*x)
sol7 = 1 + sin(x)**((exp((1/12)*(9 + (1/4)*(5 - x))))**sin(9 - x))
sol8 = (1/2)*sqrt(2)*sqrt(x) + exp(-2*x)*cos((1/2)*x) + sin(x)**(1 + sin(x)**sin(x))


sol9 = (exp(x))**(-exp(-x**3)) + sin(x)
sol10 = 2*sin(5*x)*cos(5*x)


sol11 = ((1/3)*sqrt(3)*sqrt(x))**(-2*sin((1/6)*x))
sol12 = sin(x)*sin((1/3)*x)**((exp(-x))**(sin(x)**((exp(x))**sin(x)))) + sqrt((exp((-1/12)*x))**(-exp((-1/5)*x)))
sol13 = (((1/6)*sqrt(6)*sqrt(x))**(-sin((1/8)*x)))**(5**sin(x))
sol14 = (exp((1/8)*(6 - x)))**(sin((1/9)*x)**((exp(-x))**((exp(x))**log(x))))
