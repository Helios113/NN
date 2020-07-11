import requests
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import csv
from datetime import datetime

def getPrices(duration):
    payload = {'time': duration}
    r = requests.get('https://www.bitstamp.net/api/v2/transactions/xrpeur/',params=payload)
    j = r.json()
    j.reverse()
    return j

def getStandardData(data,dataPoints,win,pT,step,a):
    sT = 0
    times =[]
    prices = []
    Ret =[]
    if a == True:
        for i in data:
            times.append(float(i['date'])/3600)
            prices.append(float(i['price']))
        f = open("data.txt", "w+")
        f1 = open("times.txt", "w+")
        np.savetxt(f, np.asarray(prices))
        np.savetxt(f1, np.asarray(times))
        f.close()
    else:
        prices = np.loadtxt("data.txt", unpack=True)
        times = np.loadtxt("times.txt",unpack = True)
    #Standartization
    #pTime = max(times)
    
    mTime = min(times)
    #return prices
    stdTime = [x - mTime for x in times]
    pTime = max(stdTime)
    normPrice = savgol_filter(prices, 51, 4)
    f = interp1d(stdTime,normPrice)
    
    while (sT+win+pT)<=pTime:
        Price = []
        for i in np.linspace(sT,sT+win,dataPoints):
            Price.append(f(i))
        
        
        mean = np.mean(Price)
        std = np.std(Price)
        m = min(Price)
        mm = max(Price)
        mm = mm-m
        Price.insert(0, f(sT+win+pT))
        stdPrice = [((x-m)/mm)-0.5 for x in Price]
        #plt.plot(stdPrice[1:])
        #plt.scatter(dataPoints*1.25,stdPrice[0])
        Ret.append(stdPrice)
        #plt.show()
        sT+=step
    return Ret

def getCSVdata():
    data = []
    with open('Bitstamp_XRPUSD_1h.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            year = int(row[0][0:4])
            month = int(row[0][5:7])
            day = int(row[0][8:10])
            tm = int(row[0][11:13])
            ampm = row[0][14:16]
            if ampm == 'PM' and tm!=12:
                tm+=12
            if ampm == 'AM' and tm ==12:
                tm =0
            dt = datetime(year,month,day,tm)
            data.append({'date':dt.timestamp(),'price':float(row[5])})
    data.reverse()
    csvfile.close()
    return data