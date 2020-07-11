import requests
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.interpolate import interp1d
from scipy import optimize
from scipy.signal import savgol_filter
import nn
#import animationTest
import time
import csv

#plt.ion()

yhat = list()
xhat = list()
ans = list()
buy = list()
initTime =0

def Sma(data, smaPeriod):
    j = next(i for i, x in enumerate(data) if x is not None)
    our_range = range(len(data))[j + smaPeriod - 1:]
    empty_list = [None] * (j + smaPeriod - 1)
    sub_result = [np.mean(data[i - smaPeriod + 1: i + 1]) for i in our_range]

    return np.array(empty_list + sub_result)

def do_something(sc):
    global yhat,xhat,ans,buy,initTime
    buy.clear()
    sell = list()
    
    payload = {'time': 'hour'}
    r = requests.get('https://www.bitstamp.net/api/v2/transactions/xrpeur/',params=payload)
    #print(r.json())
    j = r.json()
    j.reverse()

    for i in j:
        if i['type'] == '0':
            if initTime == 0:
                initTime = int(i['date'])
                print("initTime:",initTime)
            duration = (int(i['date']) - initTime)/3600
            buy.append([duration,float(i['price'])])
            lastVal = duration
              
        if i['type'] == '1':
            if initTime == 0:
                initTime = int(i['date'])
                print("initTime:",initTime)
            duration = int(i['date'])- initTime
            sell.append([duration,float(i['price'])])
    a,b = zip(*buy)
    c = a+b
    #print (n)
    
            
        
    
    xhat = [x[0] for x in buy]
    yhat = savgol_filter([x[1] for x in buy], 7, 6)
    
    fV = buy[0][0]
    lV = lastVal
    x_new1 = np.linspace(fV,lV,num = 200)
    coefs = poly.polyfit(*zip(*buy), 17)
    ffit = poly.Polynomial(coefs)
    coefss = poly.polyder(coefs)
    ffitd = poly.Polynomial(coefss)
    roots = poly.polyroots(coefss)
    intersects = np.real_if_close(roots)
    intersects = np.real(intersects)
    intersects = intersects[intersects>0]
    #print(intersects)
    f = interp1d(xhat,yhat)
    # = interp1d(*zip(*buy))
    der = np.diff(f(x_new1)) / np.diff(x_new1)
    x2 = (x_new1[1:] + x_new1[:-1]) / 2
    f1 = interp1d(x2,der)
    lastI=0
    ans =list()
    for ii in range(len(der)):
        #print (ii)
        #print(der[lastI]*der[ii])
        if der[lastI]*der[ii] <=0 and (der[lastI]!=0 or der[ii]!=0):
            #print("lasti:", lastI)
            root = optimize.ridder(f1,x2[lastI],x2[ii])
            ans.append(float(root))
        lastI=ii
    #print(ans)
    #Renderer(f,x2,f1)
    q,w = zip(*buy)
    l1 = [q,w]
    l2 = [[x[0] for x in buy],yhat]
    l3 = [x2,f1(x2)]
    #animationTest.Rend(l1, l2, l3, [])
    time.sleep(sc)
    do_something(sc)
    
    

def getStandardData(data,dataPoints):
    iTime = 0
    times =[]
    prices = []
    std =[]
    for i in data:
        if i['type'] == '0':
            if iTime ==0:
                iTime = int(i['date'])
            times.append(int(i['date']) - iTime)
            prices.append(float(i['price']))
    #Standartization
    pTime = max(times)
    pPrice = max(prices)
    mPrice = min(prices)
    pPrice = pPrice-mPrice
    stdTime = [x / pTime for x in times]
    stdPrice = [(x-mPrice)/pPrice for x in prices]
    normPrice = savgol_filter(stdPrice, 5, 4)
    f = interp1d(stdTime,normPrice)
    for i in range(dataPoints):
        std.append(float(f(i/(dataPoints-1))))
        std.append(i/(dataPoints-1))
    return std,pTime,pPrice,mPrice

def getPrices(duration):
    payload = {'time': duration}
    r = requests.get('https://www.bitstamp.net/api/v2/transactions/xrpeur/',params=payload)
    j = r.json()
    j.reverse()
    return j

def displayStdData(vals):
    a =[]
    b =[]
    for i in range(len(vals)):
        if i%2 ==0:
            a.append(vals[i])
        else:
            b.append(vals[i])
    plt.plot(b,a)
    plt.show()
    
    
    
#do_something(10)
#plt.show()
#print(datetime.utcfromtimestamp(initTime).strftime('%Y-%m-%d %H:%M:%S'))
#r1 = HighLow(buy,20)
#s1 = GetSlope(r1)
#print(r1) 

def getTrainingData(tip,data = None, window = 1,time = 0.5,step = 1/30,num_of_points = 250):
    #window - period of data (i.e. 1 hour)
    #time - what time should we predict the price for after the last input value
    #step - how much to move the window by
    #num_of_points - how many points to take from the intep1d
    sp = 0
    ep = window
    if tip == True:
        values = getPrices("day")
        values.reverse()
        times =[]
        prices = []
        for i in values:
            if i['type'] == '0' or i['type'] == '1':
                times.append(int(i['date'])/3600)
                prices.append(float(i['price']))
    else:
        times = data[0]
        prices = data[1]
    #normalization
    mTime = min(times)
    stdTime = [(x-mTime) for x in times]
    normPrice = savgol_filter(prices, 51, 1)
    f = interp1d(stdTime,normPrice)
    output = []
    mt = max(stdTime)
    rtn = []
    
    
    while(ep+time<=mt):
        p1=[]
        output =[]
        for i in np.linspace(sp,ep,num_of_points):
            p1.append(f(i))        
        pPrice = max(p1)
        mPrice = min(p1)
        pPrice = pPrice-mPrice
        stdPrice = [(x-mPrice)/pPrice for x in p1]
       
        ans = (f(ep+time)-mPrice)/pPrice
        output.append(ans)
        output.extend(stdPrice)
        sp +=step
        ep +=step
        rtn.append(output)
   # plt.show()
    return rtn
    
    
def getCSVdata():
    t = []
    prices = []
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
            t.append(dt.timestamp()/3600)
            
            prices.append(float(row[5]))
    prices.reverse()
    t.reverse()
    mt = min(t)
    t = [ x-mt for x in t]
    return t,prices
            
    