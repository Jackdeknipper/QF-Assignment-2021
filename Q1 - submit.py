

import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace

##Q1-a

T =2
#i = 1 then n=20, i=2 then n=200, i=3 then n=2000
n = 2000

times= np.linspace(0,T,n)
dt = times[1]-times[0]

cos = []

for i in range(len(times)):
        cos.append(math.cos(times[i]))

#p=1
ncos =  0 
for i in range(1,len(cos)):
    ncos += abs(cos[i]-cos[i-1])

#p=2
ncos_2 = 0
for i in range(1,len(cos)):
    ncos_2 += abs(cos[i]-cos[i-1])**2    



##############################################

#Q1-b

def Normaldistr(n,m):
    Z = np.random.normal(size=(n,m))
    return Z



def brownian(x0,T,i, m, sigma,Z):
    n = 2000
    times= np.linspace(0,T,n+1)
    dt = times[1]-times[0]


    db = np.sqrt(dt)*sigma*Z
    BB = np.full((1, m), x0)
    BM = np.concatenate((BB,np.cumsum(db,axis=0)),axis=0)

    if i == 1:
        n = 20
        for j in range(1,n+1):
            BB = np.append(BB,[BM[100*j]], axis=0)

        BM = BB
        times = np.linspace(0,T,n+1)
    
    
    if i ==2:
        n = 200
        for j in range(1,n+1):
            BB = np.append(BB,[BM[10*j]], axis=0)
        BM = BB
        times = np.linspace(0,T,n+1)




    plt.plot(times,BM)
    plt.show()


    return BM,n

Z = Normaldistr(2000,2)

a = brownian(0,2,1,1,1,Z)
b = brownian(0,2,2,1,1,Z)
c = brownian(0,2,3,2,1,Z)



def FV(BM): 
    
    fv =  np.zeros(shape=(1,BM.shape[1])) 
    for i in range(1,BM.shape[0]):
        for j in range(0,BM.shape[1]):
         fv[0][j] += abs(BM[i][j]-BM[i-1][j])
    
    
    return fv


def RV(BM): 
  
    rv =  np.zeros(shape=(1,BM.shape[1])) 
    for i in range(1,BM.shape[0]):
        for j in range(0,BM.shape[1]):
         rv[0][j] += (BM[i][j]-BM[i-1][j])**2
    
    
    return rv


#y = FV(c)
#z = RV(c)
#writeRV(y)
#writeRV(z)


##############################################
##1C


def GBM(s0, i, T, m, sigma,mu,Z):

    BM,n = brownian(0,T,i,m,1,Z)
    times= np.linspace(0,T,n+1)
    dt = times[1]-times[0]
    St = []

    for j in range(m):
        X = []
        S = []
        for i in range(len(times)):
            x = (mu-0.5*sigma**2)*times[i] + sigma*BM[i][j]
            X.append(x)
            S.append(s0*np.exp(x))
        St.append(S)

    for i in range(m):
        plt.plot(times,St[i])
    plt.show()


    return St,n,T,sigma

#functie om de data van FV in een csv te schrijven
def writeFV(y):
    filename1 = r"C:\Users\Matth\OneDrive\Documenten\University\test1.csv"
    with open(filename1,"w",encoding="utf-8-sig") as fh1:
        for i in y:
            for j in i:
                fh1.write(str(j)+"-")
    fh1.close()    


def writeRV(z):    

    filename2 = r"C:\Users\Matth\OneDrive\Documenten\University\test2.csv"
    with open(filename2,"w",encoding="utf-8-sig") as fh2:
        for i in z:
            for j in i:
                fh2.write(str(j)+"-")
    fh2.close() 


def Euler(s0, i, T, m, sigma,mu,Z):

    n = 2000
    times= np.linspace(0,T,n+1)
    dt = times[1]-times[0]


    x_total = []
    dB = Z

    for k in range(m):
        n = 2000
        X_em_small, X = [s0], s0
        for j in range(n):  
            X += mu*X*dt + sigma*X*np.sqrt(dt)*dB[j][k]
            X_em_small.append(X)

        x_total.append(X_em_small)    

        if i == 1:
            n = 20
            X_em_small = [s0]
            for j in range(1,n+1):
                X_em_small.append(x_total[k][100*j])

            x_total[k] = X_em_small
            times = np.linspace(0,T,n+1)
        
        
        if i ==2:
            n = 200
            X_em_small = [s0]
            for j in range(1,n+1):
                X_em_small.append(x_total[k][10*j])

            x_total[k] =  X_em_small
            times = np.linspace(0,T,n+1)





    for i in x_total:
        plt.plot(times, i,)

   
    plt.show()

    return x_total


S = GBM(100,3,2,2,0.2,0.1,Z)
E1 = Euler(100,3,2,2,0.2,0.1,Z)
E2 = Euler(100,2,2,2,0.2,0.1,Z)
E3 = Euler(100,1,2,2,0.2,0.1,Z)
plt.plot(linspace(0,2,2001),S[0][0], label = "Stockpath 1")
plt.plot(linspace(0,2,2001),E1[0], label = "Euler Approx.2 i=3")
plt.plot(linspace(0,2,201),E2[0], label = "Euler Approx.2 i=2")
plt.plot(linspace(0,2,21),E3[0], label = "Euler Approx.2 i=1")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncol=2, mode="expand", borderaxespad=0.)
plt.show()

plt.plot(linspace(0,2,2001),S[0][1], label = "Stockpath 2")
plt.plot(linspace(0,2,2001),E1[1], label = "Euler Approx.2 i=3")
plt.plot(linspace(0,2,201),E2[1], label = "Euler Approx.2 i=2")
plt.plot(linspace(0,2,21),E3[1], label = "Euler Approx.2 i=1")

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncol=2, mode="expand", borderaxespad=0.)
plt.show()


##############################################
#Q1-D


def RVS(S):
    

    rv =  np.zeros(shape=(1,S.shape[1])) 
    for i in range(1,S.shape[0]):
        for j in range(0,S.shape[1]):
         rv[0][j] += (S[i][j]-S[i-1][j])**2
    
    
    return rv

 
def integral(S):

    ST = S[0]
    t = S[2]/S[1]
    sigma = S[3]

    ss =  np.zeros(shape=(1,ST.shape[1])) 
    for i in range(1,ST.shape[0]+1):
        for j in range(0,ST.shape[1]):
         ss[0][j] += t*(sigma*(ST[i-1][j]))**2
    
    return ss

#i = 3
#S = GBM(100,3,2,1000,0.2,0.1,Z)
#x = RVS(S[0])
#y = integral(S[0])
#writeRV(b)
#writeFV(y)

#i = 2
#S = GBM(100,2,2,1000,0.2,0.1,Z)
#x = RVS(S[0])
#y = integral(S[0])
#writeRV(b)
#writeFV(y)

#i = 1
#S = GBM(100,2,2,1000,0.2,0.1,Z)
#x = RVS(S[0])
#y = integral(S[0])
#writeRV(b)
#writeFV(y)




