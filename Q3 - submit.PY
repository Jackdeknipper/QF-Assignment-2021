import numpy as np
from math import log,exp
from scipy.stats import norm
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
import numpy.polynomial.legendre as npleg

K = 90 #strike
S0 = 100 #Initial Stock Price
r = 0.01 #Risk Free Interest Rate
sigma = 0.15 #Volatility
Smax = 200 #Largest value of the underlying asset
Tmax = 3 #Time to Expiration in years
var = sigma**2
N = 300 #Number of time steps
M = 300 #Number of stock steps

#3a
d1 = (log(S0/K)+(r+sigma**2/2.)*Tmax)/(sigma*Tmax**0.5)
d2 = d1-sigma*Tmax**0.5
Put = K*exp(-r*Tmax)-S0+S0*norm.cdf(d1)-K*exp(-r*Tmax)*norm.cdf(d2)

print(Put)

#3b
def PDE(K,S0,r,sigma,Smax,Tmax,var,N,M):
    Time = np.linspace(0,Tmax,N)
    Stock = np.linspace(0,Smax,M)


    deltaS = Stock[1]-Stock[0]
    deltaT = Time[1]-Time[0]  



    A = np.zeros((M-2,M-2))
    for i in range(M-2):
        A[i,i] = 1 + (var*(i+1)**2+r)*deltaT
        if i-1>=0:
            A[i,i-1] = 0.5*deltaT*(r*(i+1)-var*(i+1)**2)
        if i+1<=M-3:
            A[i,i+1] = -0.5*deltaT*(r*(i+1)+var*(i+1)**2)

    V = np.zeros((N,M))
    for j in range(M):
        V[0,j] = max(K-(j+1)*deltaS,0)
        V[N-1,j] = max(K-(j+1)*deltaS,0)

    for i in range(N):
        V[i,M-1] = 0
        V[i,0] = K*np.exp(-r*((i+1)*deltaT))

    C = np.zeros((N-1,M-2))
    for i in range(N-1):
        C[i][0] = 0.5*deltaT*(r*(2)-var*(2)**2)*K*np.exp(-r*(i+1)*deltaT)
        C[i][M-3] = 0

    for k in range(N):
        curvec = V[N-1-k][1:M-1]-C[N-2-k]
        vec = np.linalg.solve(A,curvec)
        V[N-1-k-1][1:M-1] = vec

    fig = plt.figure()
    for i in range(N):
        plt.plot(Stock, V[i])
    plt.show()

#PDE(K,S0,r,sigma,Smax,Tmax,var,N,M)

##########
## 3C

def Normaldistr(n,m,sigma,T):


    times= np.linspace(0,T,n)
    dt = times[1]-times[0]
    Z = (np.sqrt(dt)*sigma*np.random.normal(size=(n-1,m)))

    return Z


def GBM(s0, n, T, m, sigma,mu,Z):



    times= np.linspace(0,T,n)
    dt = times[1]-times[0]

    ST = np.exp((mu-0.5*sigma**2)*dt+sigma*Z)
    ST = np.vstack([np.ones(m),ST])

    st= s0*ST.cumprod(axis=0)
    plt.plot(times,st)
    plt.show()


    return st,n,T,sigma   

z = Normaldistr(300,1000,1,3)
#S = GBM(100,300,3,1000,0.15,0.05,z)



def discountPut(S,K,r,Tmax):
    payoff = []
    for i in S[-1]:
        payoff.append(max(K-i,0))

    for j in range(len(payoff)):
        payoff[j] = exp(-r*Tmax)*payoff[j]

    def writeFV(y):
        filename1 = r"C:\Users\Matth\OneDrive\Documenten\University\test1.csv"
        with open(filename1,"w",encoding="utf-8-sig") as fh1:
            for i in y:
                    fh1.write(str(i)+"-")
        fh1.close() 

    writeFV(payoff)
    return payoff  

#discountPut(S[0],K,r,Tmax) 
#######
### 3D

Q = GBM(100,300,3,1000,0.15,0.01,z)
x = discountPut(Q[0],K,r,Tmax)







