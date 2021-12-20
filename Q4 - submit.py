
from math import log,exp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import savgol_filter

## 4-a:
S0 = 100
K = 90
r = 0.01
sigma = 0.2
T = 1
mu = 0.01


d = (log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*T**0.5)
delta = -norm.cdf(-d,0,1)

def BumpandReprice(n,S0,K,T,sigma,mu,r,option_type):
    h = n**(-1/4)
    Z = Normaldistr(2000,n,1,1)
    S = GBM(S0,2000,T,n,sigma,mu,Z)[0]
    S_h = GBM(S0+h,2000,T,n,sigma,mu,Z)[0] 
    X = []
    X_h = []

    for i in range(len(S[-1])):
        if option_type == "Put": 
            X.append(exp(-r*T)*max(K-S[-1][i],0)) 
            X_h.append(exp(-r*T)*max(K-S_h[-1][i],0))

    approximation = (sum(X_h)-sum(X))/(n*h)

    return approximation        
    
def Pathwise(n,S0,K,T,sigma,mu,r):
    Z = Normaldistr(200,n,1,1)
    Zi = Z.sum(axis=0)
    S = GBM(S0,200,T,n,sigma,mu,Z)[0]
    X = []

    for i in range(len(S[-1])):
        if K>S[-1][i]:
            X.append(-exp(-r*T)*exp((mu-0.5*sigma**2)*T+sigma*T**0.5*Zi[i])) 
        else:
            X.append(0)
    
    Approximation = sum(X)/len(X)

    return Approximation

def LR(n,S0,K,T,sigma,mu,r,option_type):
    Z = Normaldistr(200,n,1,1)
    Zi = Z.sum(axis=0)
    S = GBM(S0,200,T,n,sigma,mu,Z)[0]
    X = []

    for i in range(len(S[-1])):
        if option_type == "european_put":
            X.append(max(K-S[-1,i],0)*exp(-r*T)*Zi[i]/(sigma*T**0.5*S0))
        elif option_type == "digital_call":
            if S[-1,i]>K:
                X.append(exp(-r*T)*((Zi[i]**2-1)/(S0**2*sigma**2*T)-Zi[i]/(sigma*T**0.5*S0**2)))
            else:
                X.append(0)
    
    Approximation = sum(X)/len(X)
    return Approximation
    


    



#approx = BumpandReprice(10000,S0,K,T,sigma,mu,r,"Put" )
#approx2 = Pathwise(10000,S0,K,T,sigma,mu,r)
#approx3 = LR(10000,S0,K,T,sigma,mu,r,"european_put")


## 4b

def GBM(s0, n, T, m, sigma,mu,Z):

    times= np.linspace(0,T,n)
    dt = times[1]-times[0]

    ST = np.exp((mu-0.5*sigma**2)*dt+sigma*Z)
    ST = np.vstack([np.ones(m),ST])

    st= s0*ST.cumprod(axis=0)
    #plt.plot(times,st)
    #plt.show()


    return st,n,T,sigma

def Normaldistr(n,m,sigma,T):

    times= np.linspace(0,T,n)
    dt = times[1]-times[0]
    Z = (np.sqrt(dt)*sigma*np.random.normal(size=(n-1,m)))

    return Z

def Bond(B0,n,T,r):
    times= np.linspace(0,T,n)
    dt = times[1]-times[0]
    B = [B0]
    for i in range(1,n):
        B.append(B[i-1]*exp(r*dt))
    
    return B

def Putprice(S0,K,r,sigma,Tmax):
    d1 = (log(S0/K)+(r+sigma**2/2.)*Tmax)/(sigma*Tmax**0.5)
    d2 = d1-sigma*Tmax**0.5
    Put = K*exp(-r*Tmax)-S0+S0*norm.cdf(d1)-K*exp(-r*Tmax)*norm.cdf(d2)
    return Put

Z = Normaldistr(2000,1,1,1)
S = GBM(100,2000,1,1,0.2,0.06,Z)
B = Bond(1,2000,1,0.01)

def replicating(S,B,K,r,sigma,T):

    Ct = []
    Kt = []
    for i in range(len(S)):
        t = T/len(S)
        ti = t*i
        Ct.append(Putprice(S[i],K,r,sigma,(T-t*i)))
        Kt.append(K)

    phi = []
    psi = []
    for j in range(1,len(S)):
        dV = Ct[j]-Ct[j-1]
        dS = S[j]-S[j-1]
        phi.append(dV/dS*S[j])
        psi.append(Ct[j]-phi[j-1])

    #Smoothing the graph in order to make it readable:
    phi_hat = savgol_filter(phi,199,4)   
    psi_hat = savgol_filter(psi,199,4) 
   
    times= np.linspace(0,T,len(S))
    dt = times[1]-times[0]
    
    plt.plot(times,S, label = "Stockpath")
    plt.plot(times,Kt, label = "Strike")
    plt.plot(times[1:len(times)],phi_hat, label = "Stock position")
    plt.plot(times[1:len(times)],psi_hat, label = "Bond position")
    plt.plot(times,Ct,label = "Option price")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncol=3, mode="expand", borderaxespad=0.)
    plt.show()
    return phi,psi,Ct




def multi_replicating(m,S,B,K,r,sigma,T): #in order to replicate 5 portfolio's simultaneously, however we decided to plot 5 seprate graph's, because it's hard to read the graphs otherwise.
    num_rows, num_cols = S[0].shape
    times= np.linspace(0,T,int(num_rows))
    dt = times[1]-times[0]
    colors = ["b","g",'r',"c","m"]


    for i in range(0,m):
        x = replicating(S[0][:,i],B,90,0.01,0.2,1)
        plt.plot(times,S[0][:,i], label = "Stockpath {line}".format(line = str(i+1)),color = colors[i])
        plt.plot(times[1:len(times)],x[0], label = "Stock position {line}".format(line = str(i+1)), color =colors[i],linestyle='dashed')
        plt.plot(times[1:len(times)],x[1], label = "Bond position {line}".format(line = str(i+1)), color = colors[i],linestyle = 'dotted')

    plt.plot(times,x[2])
    plt.legend(bbox_to_anchor=(1,1),loc ="upper left")

    plt.show()

#run 5 times:
#multi_replicating(1,S,B,90,0.01,0.2,1)
    
##################################################
#4-c

def deltahedging(T,S,K,S0,r,sigma,B):
    times= np.linspace(0,T,int(T/0.02))

    C0 = 1000*Putprice(S0,K,r,sigma,T)
    d = (log(S0/K)+(r+sigma**2/2.)*T)/(sigma*T**0.5)
    phi0 = 1000*-norm.cdf(-d)
    psi0 = C0 -phi0*S0

    for i in range(1,len(times)-1):
        d = (log(S[40*i]/K)+(r+sigma**2/2.)*(T-times[i])/(sigma*(T-times[i])**0.5))
        phi = 1000*-norm.cdf(-d)
        psi = (C0-phi*S[40*i])/B[i]
    
    portfolio = phi*S[-1]+psi*B[-1]
    Ct = max(K-S[-1],0)
    payoff = portfolio-1000*Ct


    return payoff

def simulate_deltahedge(n):
    Z = Normaldistr(2001,n,1,1)
    S = GBM(100,2001,1,n,0.2,0.06,Z)
    B = Bond(1,2001,1,0.01)
    payoffs = []
    for i in range(n):
           payoffs.append(deltahedging(1,S[0][:,i],90,100,0.01,0.2,B)) 

    x = sum(payoffs)/len(payoffs)
    return payoffs


#x = simulate_deltahedge(2500)


def writeFV(y): #write our simulations to excel.
    filename1 = r"C:\Users\Matth\OneDrive\Documenten\University\test1.csv"
    with open(filename1,"w",encoding="utf-8-sig") as fh1:
        for i in y:
            fh1.write(str(i)+"/")
    fh1.close() 

#writeFV(x)

##########################################
## 4-D

def Digital_Callprice(S0,K,r,sigma,Tmax):
    d = (log(S0/K)+(r-sigma**2/2.)*Tmax)/(sigma*Tmax**0.5)
    Digital_Call = exp(-r*Tmax)*norm.cdf(d)
    return Digital_Call


def gammahedging(T_1,S,K_1,S0,r,sigma,B,T_2,K_2):
    times= np.linspace(0,T_1,int(T_1/0.02))

    C0 = 1000*Putprice(S0,K_1,r,sigma,T_1)
    dp = (log(S0/K_1)+(r+sigma**2/2.)*T_1)/(sigma*T_1**0.5)
    dc = (log(S0/K_2)+(r-sigma**2/2.)*T_2)/(sigma*T_2**0.5)
    gamma_put = 1/(S0*sigma*T_1**0.5)*norm.pdf(dp)
    delta_call = exp(-r*T_2)/(S0*sigma*T_2**0.5)*norm.pdf(dc)
    gamma_call = LR(10000,S0,K_2,T_2,sigma,r,r,"digital_call")
    P0 = Digital_Callprice(S0,K_2,r,sigma,T_2)
    
    omega = 1000*gamma_put/gamma_call

    phi0 = 1000*-norm.cdf(dp)-omega*delta_call
    psi0 = C0 -phi0*S0-omega*P0

    for i in range(1,len(times)-1):
        dp = (log(S[40*i]/K_1)+(r+sigma**2/2.)*(T_1-times[i])/(sigma*(T_1-times[i])**0.5))
        dc = (log(S[40*i]/K_2)+(r+sigma**2/2.)*(T_2-times[i])/(sigma*(T_2-times[i])**0.5))


        gamma_put = 1/(S[40*i]*sigma*(T_1-times[i])**0.5)*norm.pdf(dp)
        delta_call = exp(-r*(T_2-times[i]))/(S[40*i]*sigma*(T_2-times[i])**0.5)*norm.pdf(dc)
        gamma_call = LR(10000,S[40*i],K_2,(T_2-times[i]),sigma,r,r,"digital_call")
        omega = 1000*gamma_put/gamma_call

        phi = 1000*-norm.cdf(dp)-omega*delta_call
        psi = C0 -phi*S[40*i]-omega*Digital_Callprice(S[40*i],K_2,r,sigma,T_2-times[i])
    
    portfolio = phi*S[-1]+psi*B[-1]+omega*Digital_Callprice(S[-1],K_2,r,sigma,T_2-T_1)
    Ct = max(K-S[-1],0)
    payoff = portfolio-1000*Ct


    return payoff


def simulate_gammahedge(n):
    Z = Normaldistr(2001,n,1,1)
    S = GBM(100,2001,1,n,0.2,0.06,Z)
    B = Bond(1,2001,1,0.01)
    payoffs = []
    for i in range(n):
           payoffs.append(gammahedging(1,S[0][:,i],90,100,0.01,0.2,B,2,120)) 

    x = sum(payoffs)/len(payoffs)
    return payoffs


y = simulate_gammahedge(100)
writeFV(y)


