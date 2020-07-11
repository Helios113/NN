import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

prices = sorted((np.random.rand(5000)-0.5)/np.sqrt(5000))
prices1 = prices

#raw data 1
mean = np.mean(prices)
std = np.std(prices)
fit = stats.norm.pdf(prices,mean,std)
plt.plot(prices,fit,'--')
plt.hist(prices)
#standartization
prices = [(i-mean)/std for i in prices]
mean = np.mean(prices)
std = np.std(prices)
fit = stats.norm.pdf(prices,mean,std)
#plt.hist(prices, density = True)
#plt.plot(prices,fit,'--')
#normalization
pp = max(prices)
pm = min(prices)
pp = pp-pm
prices = [((i-pm)/pp)-0.5 for i in prices]
mean = np.mean(prices)
std = np.std(prices)
fit = stats.norm.pdf(prices,mean,std)
#plt.hist(prices, density = True)
#plt.plot(prices,fit,'--')


#normalization of non standart data
pp = max(prices1)
pm = min(prices1)
pp = pp-pm
prices1 = [(i-pm)/pp for i in prices1]
mean1 = np.mean(prices1)
std1 = np.std(prices1)
fit = stats.norm.pdf(prices1,mean1,std1)
#plt.plot(prices1,fit,'--')


# standartization of normal standart data
prices = [(i-mean)/std for i in prices]
mean = np.mean(prices)
std = np.std(prices)
fit = stats.norm.pdf(prices,mean,std)
#plt.plot(prices,fit,'--')

#standartization of normal data
prices1 = [(i-mean1)/std1 for i in prices1]
mean = np.mean(prices1)
std = np.std(prices1)
fit = stats.norm.pdf(prices1,mean,std)
#plt.plot(prices1,fit,'--')

plt.show()