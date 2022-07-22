#!/usr/bin/env python3
{
   "shell_cmd": "gnome-terminal -- bash -c \"python3 -u $file;echo;echo Press Enter to exit...;read\"",
   "file_regex": "^[ ]*File \"(...*?)\", line ([0-9]*)",
   "selector": "source.python",
   
}

from numpy.random import random, normal, choice
from sympy import *
import sys
import math
import numpy

x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

#E = ["EOE","(E)","F(E)","D","x","y","z"] # mod 7
E = ["EOE","(E)","F(E)","D","x"] # mod 5
##     0     1      2    3   4   5   6
O = ["+","-","*","/"] # mod 4
##    0   1   2   3
F = ["sin","cos","exp","log"] # mod 4
##     0     1     2     3
D = ["0","1","2","3","4","5","6","7","8","9"] # mod 10

#V = ["x","y", "z"]



dict = {}
dict["E"] = E
dict["O"] = O
dict["F"] = F
dict["D"] = D

def mayus(s):
    res = 0
    for si in s:
        if si.isupper():
            res = 1
            break
    return res

def decode(cromosoma):
    string = ""
    estado = E
    dale = 0
    i = 0
    cromos = []
    letra = "E"
    for c in cromosoma:
        cromos.append([letra, c%len(estado)])
        if i!=0 and not mayus(string):
            #print("ya paro")
            break
        #print("voy con ",c, "en estado", estado)
        if dale:
            string= string[:i] + estado[c%len(estado)] + string[i+1:]
            i=0
        else:
            string+=(estado[c%len(estado)])
        #print("ASI VA: ", string)
        for s in string:
            #print("s: ",s)
            if s.isupper():
                #print("encontre mayuscula ", i)
                letra = s
                estado = dict[s]
                dale = 1
                break
            else:
                i+=1
                dale = 0
    #string =  simplify(string)
    """
    try:
        string = simplify(string)
    except:
        return -1
    """
    return [string,cromos[:]]
    #return sympify(string)

"""
cromosome = [16, 3, 7, 4, 10, 28, 24, 1,2,4]
print(decode(cromosome))
"""



def c0(N):
    ci = []
    ci.append(int(random()*8))
    for k in range(N):
        ci.append(int(255*random()))
        """
    sigue = 1
    while sigue:
        if not mayus(decode(ci)):
            break
        ci.append(int(10*random()))
        """
    
    return ci

def limpia(s):
    res = ""
    res = res.join(s) 
    j = 0
    for i in range(len(s)):
        if (i<len(s)-1):
            str1 = ""
            str2 = ""
            str1 = str1.join(s[i])
            str2 = str2.join(s[i+1])
            if s[i]==')' and s[i+1]=='(':
                #print("akiii es ",s[i], s[i+1])
                res = res[:j+1] + "*" + res[j+1:]
                j+=1
            elif str1.isdigit() and not str2.isdigit() and s[i+1]!=')':
                #print("aki es ",str1, str2)
                res = res[:j+1] + "*" + res[j+1:]
                j+=1
        j+=1
    return res



def crea():
    ci = []
    ci.append(int(len(E)*random()))
    j=0
    while not mayus(decode(ci)):
        ci.append(int(255*random()))
        j+=1
    print(j)
    return ci



def fitness(f):
    f = sympify(f)
    ecuacion =  abs(f.diff(x) - f) 
    pvi = 2*abs(y0-f.subs({x : x0})) 
    return ecuacion+pvi


from sympy.printing.theanocode import theano_function

def fit(f):
    val = 0
    #data = numpy.linspace(a,b,10)
    #print(data)
    #fu = lambdify(x,fitness(f))
    #arr = fu(data)
    #print(arr)
    #val = arr.sum()
    #val = sum(arr)
    #val = arr
    
    #f = theano_function([x], [f])
    #print("aki")
    try:
        fu = lambdify(x,fitness(f))
    except:
        return 9999999
        
    
    #print("oaki")
    for xkk in xk:
        
        #fu = lambdify(x,fitness(f))
        #val+=round(fitness(f).subs({x:xkk}),precision)
        try:
            #f = theano_function([x], [f], dims={x: xkk}, dtypes={x: 'float64'})
            #print("f de ",xkk,fu(xkk))
            val+=abs(round(fu(xkk),precision))
        except:
            return 9999999
    if math.isnan(val):
        return 999999
    return val

def eval(M):
    V = []
    for persona in M:
        print("voy con ", persona[0])
        V.append([persona, round(fit(persona[0]),precision)])
    return V


def selecciona(V):
    # selecciona
    res = []
    mejores = [] 
    #valormejor = []
    for k in range(int(len(V)/2)):
        index = min(V)
        mejores.append(V[index][0])
        #valormejor.append(V[index][1])
        res.append([V[index][0],V[index][1]])
        V.pop(index)

    #print(" valor mejores ",mejores)
    #return res
    #print("los duros ",res)
    return mejores

def min(x):
    mini = 0
    for i in range(len(x)):
        if math.isnan(x[i][1]):
            continue
        if x[i][1]<x[mini][1]:
            mini = i
    return mini

def contenido(x,y):
    res = 0
    for yi in y:
        if x==yi:
            res = 1
            break
    return res

def muta(persona, conca):
    cromosomas = persona[1]
    cromo = []
    sec = []
    algo = 0
    for c in cromosomas:
        #print("par ",c)
        #print("letra ",c[0])
        if c[0]=='F':
            c[1]=int(len(F)*random())
            algo = 1
        if c[0]=='O':
            c[1]=int(len(O)*random())
            algo = 1
        if c[0]=='D':
            c[1]=int(len(D)*random())
            algo = 1
        
    
        sec.append(c[1])
        cromo.append(c)
    if conca==1:
        #print("es simple ",persona[0])
        sec.insert(0,0) # EOE
        posible = nuevo()
        while posible==-1:
            print("nuevo fresh")
            posible = nuevo()
        trozo = []
        for k in posible[1]:
            trozo.append(k[1])
        sec = sec+trozo
    #print("asi keda ",cromo)
    return decode(sec)

import time

def cruza(mejores, M):
    res = []
    for m in mejores:
        #res.append(sympify(m))
        #res.append(sympify(muta(m)))
        res.append(m)
        nuevo = muta(m,0)
        new = sympify(nuevo[0])
        while contenido(new,M)==1:
            print("nito fresh")
            #print("CONCATENA ")
            #print("stuck, ",new," ta en ", M)
            #time.sleep(2)
            nuevo = muta(m,1)
            new = sympify(nuevo[0])
        M.append(new)
        res.append([new, nuevo[1]])
        #res.append(muta(m))
    return res

x0, y0 = 0.1,2.1/(sin(0.1)) # condicion inicial
Dominio = 10
[a,b] = [x0, x0+Dominio] # Dominio de t
h = Dominio/10 # discretizacion 

precision = 5


# discretizacion
values = [{ x : round(a+i*h,precision) } for i in range(int((b-a)/h)+1)]
xk = [round(a+i*h,precision)  for i in range(int((b-a)/h)+1)]



M = []
p = 100 #poblacion
N = 5  #cromosomas
mundo = []

def nuevo():
    kk = decode(c0(N))
    strin = kk[0]
    c = kk[1]
    if mayus(strin) == 0 and strin!=sympify("nan") and strin!=sympify("zoo"):
        strin = sympify(strin)
        if contenido(strin,M)==0 :
            return [strin,c]
    return -1
            
while len(mundo)<p:
    posible = nuevo()
    if posible!=-1:
        M.append(posible[0])
        mundo.append([posible[0],posible[1]])
            
    #print(strin)

for k in range(100):
    print(len(M))
    print()
    #print(" mundo ", mundo)
    V = eval(mundo)
    #print("evaluado ", V)
    
    mejor = V[min(V)]
    print("ganador: ",mejor)
    
    mejores = selecciona(V)
    #print("mejores ",mejores)
    
    mundo = cruza(mejores, M)



V = eval(mundo)

mejor = V[min(V)]

print("ganador final: ",mejor)


"""
[0,3,1,3,0,2,2,3,1,2,2,2,4] = 1/exp(1)*exp(x)
[0,3,1,3,2,2,3,1] = 1/exp(1)
[2,2,4] = exp(x)

['1/exp(1)*exp(x)',
 [['E', 0],
  ['E', 3],
  ['D', 1],
  ['O', 3],
  ['E', 0],
  ['E', 2],
  ['F', 2],
  ['E', 3],
  ['D', 1],
  ['O', 2],
  ['E', 2],
  ['F', 2],
  ['E', 4]]]

['1/exp(1)',
 [['E', 0],
  ['E', 3],
  ['D', 1],
  ['O', 3],
  ['E', 2],
  ['F', 2],
  ['E', 3],
  ['D', 1]]]

['exp(x)', [['E', 2], ['F', 2], ['E', 4]]]
"""









