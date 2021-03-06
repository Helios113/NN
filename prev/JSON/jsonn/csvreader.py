import csv
import os
pth = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy import interpolate

def FileGetter():
    ls = []
    with open(pth+'/Kraken_XRPUSD_1h.csv', 'r',newline='') as price_data:
        price_reader = csv.reader(price_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in price_reader:
            ls.append([row[0],row[3],row[4]])
    return ls


def PreslavIndicator(data):
    maxTime = data[0][0]
    minTime = data[-1][0]
    ls1 = [[i[0]-minTime,i[1]] for i in data if i[0]-minTime >=0.8*(maxTime-minTime)]
    ls2 = [[i[0]-minTime,i[1]] for i in data]
    ls1.reverse()
    ls2.reverse()
    a1,a2 = zip(*ls1)
    a3,a4 = zip(*ls2)
    Area1 = np.trapz(a2, a1)
    Area2 = np.trapz(a4, a3)
    return ((Area1/(0.2*(maxTime-minTime)))-(Area2/(maxTime-minTime)))






a,b,c = zip(*FileGetter())




n =4 #number of hours to combine
per = 34

a = [datetime.strptime(i, '%Y-%m-%d %I-%p').timestamp() for i in a[2:]]
b = [float(i) for i in b[2:]]
c = [float(i) for i in c[2:]]
d = [sum(x)/2 for x in zip(b, c)]
e = [sum(d[i:i+n])/n for i in range(0,len(d),n)]
f = [sum(a[i:i+n])/n for i in range(0,len(a),n)]
dt = list(zip(f,e))
wws =[]
for i in range(0,len(e)-per):
    wws.append(PreslavIndicator(dt[i:i+per]))
svg = savgol_filter(wws, 31, 5)
x = f[:len(f)-per]
y = e[:len(f)-per]
#dy = np.zeros(svg.shape,np.float)
#dy[0:-1] = np.diff(svg)
#dy[-1] = (svg[-1] - svg[-2])
#dx = np.zeros(svg.shape,np.float)
#dx[0:-1] = np.diff(f[:-per])
#dx[-1] = (f[-per] - svg[-(per+1)])
#dx/=50000
#dy=dy/dx
dy = np.gradient(svg,x)
dy*=20000
dy1 = np.gradient(dy,x)
dy1*=90000







peaks,_ = find_peaks(svg,prominence = 0.02)
valeys,_ = find_peaks(-svg,prominence = 0.02)
peaks = peaks.tolist()
valeys = valeys.tolist()


"""
test
"""

e = np.asarray(e)
peaks1,_ = find_peaks(e[:len(e)-per],prominence = 0.08)
#valeys1,_ = find_peaks(-e[:len(e)-per],prominence = 0.08,distance = 80)
#peaks1 = peaks1.tolist()
#valeys1 = valeys1.tolist()
#peaks1.extend(valeys1)
#peaks1.sort()

"""
solutions =[]
for i in range(len(peaks1)):
    col = 'm'
    #plt.scatter(f[peaks1[i]],e[peaks1[i]],c = col)
    #plt.scatter(f[peaks1[i]],dy[peaks1[i]],c = col)
    #plt.scatter(f[peaks1[i]],svg[peaks1[i]],c = col)
    solutions.append(dy1[peaks1[i]])
m = np.mean(solutions)
u = np.std(solutions)
print(m,u)
#plt.hist(solutions)
#plt.show()
"""
x, dy1,y,svg = (list(t) for t in zip(*sorted(zip(x, dy1,y,svg))))
spl = interpolate.InterpolatedUnivariateSpline(x, svg)
der = spl.derivative(1)



#plt.plot(x,wws)
#plt.plot(f[:len(f)-per],svg)
#plt.plot(x,dy)
#plt.scatter(zeros,ys)
plt.plot(x,svg)
plt.plot(x,y)
#for i in range(len(peaks)):
#    plt.scatter(f[peaks[i]],e[peaks[i]])
#    plt.scatter(f[peaks[i]],svg[peaks[i]])
plt.show()