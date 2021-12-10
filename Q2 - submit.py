#!/usr/bin/env python
# coding: utf-8

# In[180]:


#assignment exercise 2a
from datetime import datetime
#first we download the file from https://live.euronext.com/en/product/indices/NL0000000107-XAMS
#and form the a list with the AEX values
file = open('AEX-INDEX_1.csv', 'r') 
a = file.readlines()
del a[0:4] 
AEX = []
dates = [] #we also form a list with the corresponding dates of the AEX values 
for line in a:
    b = line.split(";")
    AEX.append(float(b[1]))
    dates.append(b[0])
    
#Next we form a list dt containing the difference in date in days between each two succesive AEX values
date_format = "%m/%d/%Y"
dt=[]
for i in range(len(dates)-1):
    a = datetime.strptime(dates[i], date_format)
    b = datetime.strptime(dates[i+1], date_format)
    dt.append((a-b).days)
dt[42] = 1

#Then we also create a list AEXdif with the difference between two succesive AEX values
AEXdif = []
for i in range(len(AEX)-1):
    AEXdif.append(AEX[i+1]-AEX[i])
    
#Next we use that dXt = Xt(mu*dt + sigma*dW), so (mu*dt + sigma*dW) = dXt/Xt
variable = [] #variable will contain the values (mu*dt + sigma*dW)
for i in range(len(AEXdif)):
    variable.append(AEXdif[i]/float(AEX[i]))
newvariable = [] #We create newvariable which will take the different dt's into account
for i in range(len(AEXdif)):
    newvariable.append(AEXdif[i]/(float(AEX[i])*dt[i]))
mu = sum(newvariable)/len(newvariable) #Since E[sigma*dW] = 0, mean=mu*dt. As we already took dt into account in the previous step we get mu
mean = sum(variable)/len(variable)
var = sum((x-mean)**2 for x in variable) / (len(variable)-1) #this is equal to sigma
print(mu, var)


# In[182]:


#2b
import numpy as np

def black_scholes_call(S, K, r, sigma, T):
    d = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    price = np.exp(-r * T) * norm.cdf(d)
    return price
S = AEX[0]
r = 0.01 
total_price = np.exp(-r*5)*100 
for T in range(1,6):
    total_price += black_scholes_call(S, 850, r, sigma, T)
print(total_price)

