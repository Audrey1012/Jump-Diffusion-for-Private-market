#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd

def merton_jump_paths(S, T, r, sigma,  lam, m, v, steps, Npaths):
   size=(steps,Npaths)
   dt = T/steps 
   poi_rv = np.multiply(np.random.poisson( lam*dt, size=size),
                        np.random.normal(m,v, size=size)).cumsum(axis=0)
   geo = np.cumsum(((r -  sigma**2/2 -lam*(m  + v**2*0.5))*dt +                             sigma*np.sqrt(dt) *                              np.random.normal(size=size)), axis=0)
   
   return np.exp(geo+poi_rv)*S


S = 100 # current stock price
T = 1 # time to maturity
r = 0.02 # risk free rate
m = 0 # meean of jump size
v = 0.3 # standard deviation of jump
lam =1 # intensity of jump i.e. number of jumps per annum
steps =10000 # time steps
Npaths = 1 # number of paths to simulate
sigma = 0.2 # annaul standard deviation , for weiner process

j = merton_jump_paths(S, T, r, sigma, lam, m, v, steps, Npaths)

plt.plot(j)
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Jump Diffusion Process')


# In[63]:


df = pd.read_csv('C:/Users/saxen/Downloads/AFP_Data.csv')
df['date'] = pd.to_datetime(df['date'])
from datetime import timedelta
df['date'].iloc[0] + timedelta(days=1) 
end_date = df['date'][len(df['date'])-1]
str_date = df['date'][0]



df['date_count'] = df.date.map(df.groupby('date').size())
count= sum(df['date_count'])

event_dates = len(df['date'].unique())


IPO_count=0 

bankruptcy_count = 0

for i in range(len(df)):
    if df['deal_type'][i] == "IPO":
        IPO_count +=1 
IPO_count  

for j in range(len(df)):
    if df['deal_type'][j] == "Bankruptcy: Liquidation":
        bankruptcy_count +=1 
        
for k in range(len(df)):
    if df['deal_type'][k] == "Bankruptcy: Admin/Reorg":
        bankruptcy_count +=1        

        
total_dates =0

t = str_date
while t < end_date:
    total_dates += 1
    t += timedelta(days=1)

No_round_count = total_dates - event_dates

all_dates_count = total_dates

round_count = event_dates - IPO_count - bankruptcy_count


#############

round_prob = round_count/ all_dates_count
IPO_prob = IPO_count/ all_dates_count
bankruptcy_prob = bankruptcy_count/all_dates_count
no_round_prob = No_round_count/all_dates_count

print(round_prob,IPO_prob, no_round_prob,bankruptcy_prob)


# In[91]:
result = pd.DataFrame()
f = {"company_id": [0],"post_money_valuation": [0], "volatility": [0], "mean_return": [0],"option_price": [0]}
result = pd.DataFrame(f)
for q in range(7,101):
    
    df1 = pd.read_csv('C:/Users/saxen/Downloads/Company/company'+str(q)+'.csv')
    df1['date'] = pd.to_datetime(df1['date'])
    
    if len(df1)<=2:
        continue
    
    a=0.007
    b=1
    
    df1["running_prob"] = 1/(1 + np.exp(-a*(np.log(df1['post_money_valuation']) - b )))      
            
    c_prob_round = df1["running_prob"][len(df1)-1]
    c_prob_IPO = IPO_prob
    c_prob_bankruptcy = bankruptcy_prob
    c_no_round_prob = 1-c_prob_round -c_prob_IPO-c_prob_bankruptcy
    
    print(c_prob_round,c_prob_IPO,c_no_round_prob, c_prob_bankruptcy)
    
    
    # In[92]:
    
    
    data = pd.read_csv('C:/Users/saxen/Downloads/Company/company'+str(q)+'.csv')
    data['date'] = pd.to_datetime(data['date'])
    
    if len(data)<=2:
        continue
    
    # In[93]:
    
    
    lambada = []
    for i in range(1,len(data)-1):
        jump=0
        jump = (data["post_money_valuation"][i] - data['deal_amount'][i] - data["post_money_valuation"][i-1])/data["post_money_valuation"][i-1]
        time = (data['date'][i] - data['date'][i-1]).days
        lambada.append(jump*365/time)
    
    print(np.mean(lambada))
    print(np.std(lambada))
    intensity = len(lambada)/(((data['date'][len(data)-1] - data['date'][0]).days)/365)
    print(intensity)
    
    
    # In[94]:
    
    
    geom = []
    for i in range(1,len(data)-1):
        ret=0
        ret = (data["deal_amount"][i])/data["post_money_valuation"][i-1]
        time = (data['date'][i] - data['date'][i-1]).days
        geom.append(ret*365/time)
    geom= geom[1:]
    geom
    
    
    # In[95]:
    
    
    S = data['post_money_valuation'][0] # current stock price
    T = ((data['date'][len(data)-1] - data['date'][0]).days)/365 # time to maturity
    
    ##Selection bias removal
    c_prob_round
    c_prob_IPO
    c_prob_bankruptcy
    c_no_round_prob
    
    r = np.mean(geom) # risk free rate
    m = np.mean(np.array(lambada)*np.array(c_prob_IPO)) # meean of jump size
    v = np.std(np.array(lambada)*np.array(c_prob_IPO)) # standard deviation of jump
    lam = intensity # intensity of jump i.e. number of jumps per annum
    steps =  3825 #time steps
    Npaths = 10000 # number of paths to simulate
    sigma = np.std(geom) # annaul standard deviation , for weiner process
    
    j = merton_jump_paths(S, T, r, sigma, lam, m, v, steps, Npaths)
    
    plt.plot(np.mean(j,axis=1))
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.title('Jump Diffusion Process')
    
    
    # In[96]:
    
    
    plt.plot(data['post_money_valuation'])
    
    
    # Call Option
    
    # In[97]:
    
    
    K = 500000000
    rf=0.02
    mcprice = np.maximum(np.mean(j,axis=1)[-1]-K,0)*np.exp(-rf*T)
    mcprice
    
    
    # Simulation
    
    # In[98]:
    
    
    S = data['post_money_valuation'][len(data)-1] # current stock price
    T = 5 # time to maturity
    r = np.mean(geom) # risk free rate
    m = np.mean(lambada) # meean of jump size
    v = np.std(lambada) # standard deviation of jump
    lam = intensity # intensity of jump i.e. number of jumps per annum
    steps =  1275 #time steps
    Npaths = 10000 # number of paths to simulate
    sigma = np.std(geom) # annaul standard deviation , for weiner process
    
    j_sim = merton_jump_paths(S, T, r, sigma, lam, m, v, steps, Npaths)
    plt.plot(np.mean(j_sim,axis=1))
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.title('Simulated price')
    
    
    # Liquidation method
    
    # In[99]:
    
    
    theta=0.5
    V_thresh = data['post_money_valuation'][len(data)-1]
    u1 = (sigma**2/2 - r - lam*np.log(theta) + np.sqrt((sigma**2/2 - r - lam*np.log(theta))**2 +(2*sigma**2+2*lam*(np.log(theta))**2)*(lam+rf)))/(sigma**2+lam*(np.log(theta))**2)
    a1= (1/u1)*(V_thresh**(1-u1))
    
    
    # In[100]:
    
    
    V = np.mean(j,axis=1)[-1]
    Liq_val = a1*(V**u1)
    Liq_val
    
    
    # In[101]:
    
    
    ##Assuming K = liquidation value
    mcprice = np.maximum(np.mean(j,axis=1)[-1]-Liq_val,0)*np.exp(-rf*T)
    mcprice
    
    
    # In[ ]:
    
    
    f = {"company_id": [str(q)],"post_money_valuation": [data['post_money_valuation'][len(data)-1]], "volatility": [v], "mean_return": [m],"option_price": [mcprice]}
    dk = pd.DataFrame(f)
    result = result.append(dk, ignore_index= True)
result.to_csv('C:/Users/saxen/Downloads/Result/Result_final1.csv')