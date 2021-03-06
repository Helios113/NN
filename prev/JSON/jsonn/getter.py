import requests
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from datetime import datetime
import time
import FileHandler as fh


def getTransactions(duration):
    payload = {'time': duration}
    r = requests.get('https://www.bitstamp.net/api/v2/transactions/xrpeur/',params=payload)
    j = r.json()
    return j


def getOrders():
    r = requests.get('https://www.bitstamp.net/api/v2/order_book/xrpeur/')
    j = r.json()
    a = j['bids']
    b = j['asks']
    list1 =[]
    list2 =[]
    for i in a:
        list1.append(list(map(float,i)))
    for i in b:
        list2.append(list(map(float,i)))
    return list1,list2


def SupplyDemandFactor(_bids,_asks):
    bc = 0.85*_bids[0][0]
    ac = 1.15*_asks[0][0]
    wlb = [i for i in _bids if i[0] >=bc]
    wlb.reverse()
    wla = [i for i in _asks if i[0] <=ac]
    wl1,wl2 = zip(*wlb)
#    plt.plot(wl1,wl2)
    bidArea = np.trapz(wl2, wl1)
    wl1,wl2 = zip(*wla) 
#    plt.plot(wl1,wl2)
    askArea = np.trapz(wl2, wl1)
#    plt.show()
    return bidArea/askArea

"""
Breaks data into specified periods used in the Awesome Ocsilator
"""
def BreakIntoPeriod(period,data):
    initialTime = data[0][0]
    print("Initial time is: ", initialTime)
    finalTime= initialTime-34*period
    print("Final time is: ", finalTime)
    if finalTime<0:
        print(initialTime)
        print("Oops!  Period too high.  Try again...")
        raise ValueError
        return
    currentTime = initialTime-period
    wl=[]
    while currentTime>=finalTime:
        p = [i[1] for i in data if i[0] >=currentTime]
        wl.append((max(p)+min(p))/2)
        currentTime-= period
    print(len(wl))
    return wl

def AO(period,data):
    wl=[]
    minTime = float(data[-1]['date'])/3600
    maxTime = float(data[0]['date'])/3600
    print("Min Time:",minTime)
    print("Max Time:",maxTime)
    for i in data:
        wl.append([(float(i['date'])/3600)-minTime,float(i['price'])])
    ao = BreakIntoPeriod(period, wl)
    print(len(ao[0:5]))
    print(ao)
    val = (sum(ao[0:5])/5)-(sum(ao)/34)
    
def PreslavIndicator(data):
    maxTime = float(data[0]['date'])/3600
    minTime = float(data[-1]['date'])/3600
    ls1 = [i for i in data if (float(i['date'])/3600)-minTime >=0.8*(maxTime-minTime)]
    ls2 = data
    wl1=[]
    wl2=[]
    for i in ls1:
        wl1.append([(float(i['date'])/3600)-minTime,float(i['price'])])
    for i in ls2:
        wl2.append([(float(i['date'])/3600)-minTime,float(i['price'])])
    a1,a2 = zip(*wl1)
    a3,a4 = zip(*wl2)
    
    Area1 = np.trapz(a2, a1)
    Area2 = np.trapz(a4, a3)
    return ((Area1/4.8)-(Area2/24))
    

def Funk(b):
    bids,asks = getOrders()
    sdf = SupplyDemandFactor(bids, asks)
    d = getTransactions("day")
    ind = PreslavIndicator(d)
    fh.FileAdder([sdf,ind,time.time(),b[2]], '/indictors.csv')