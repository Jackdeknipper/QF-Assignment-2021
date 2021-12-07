#!/usr/bin/env python
# coding: utf-8

import numpy as np 
from scipy.stats import norm
from math import log

#assignment exercise 2a
import csv
file = open('AEX-INDEX_quote_chart.csv')
#file = open('AEX-INDEX_Historical_price.csv')
reader = csv.reader(file, quoting=csv.QUOTE_ALL, skipinitialspace=True)
AEX = [] #not integer!. First of all a list 'AEX' is created with the AEX index values that we retrieved
for line in reader:
    AEX.append(line[1])
#The header and the first value which is from a different date  
#compared to the other values (all a minute apart from each other) are deleted from 'AEX'
del AEX[0:2]
#next the differences in AEX index for each step are stored in AEXdif
AEXdif = []
for i in range(len(AEX)-1):
    AEXdif.append(float(AEX[i+1])-float(AEX[i]))
#dXt = Xt(mu*dt + sigma*dW), so (mu*dt + sigma*dW) = dXt/Xt
variable = []
for i in range(len(AEXdif)):
    variable.append(AEXdif[i]/float(AEX[i]))
mean = sum(variable) #Since E[sigma*dW] = 0, mean=mu*dt=mu
var = sum((x-mean)**2 for x in variable) / (len(variable)-1) #this is equal to sigma**4
sigma = np.sqrt(var)
mu = mean + var/2
print(mu, sigma)


# In[88]:


#2b


def black_scholes_call(S, K, r, sigma, T):
    d = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    price = np.exp(-r * T) * norm.cdf(d)
    return price
S = float(AEX[0])
r = .01
total_price = np.exp(-r*5)*100 
for T in range(1,6):
    total_price += black_scholes_call(S, 850, r, sigma, T)
print(total_price)


# In[89]:





# In[ ]:

