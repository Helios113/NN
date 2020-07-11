import numpy as np
import Neural as nu
import testies as ts
import matplotlib.pyplot as plt
inputVals = ts.getTrainingData(True,step = 1/90,num_of_points = 300)
#print(inputVals)
#matrix char
w = 3
h = len(inputVals[0])-1
lr = 0.01

#Matrix init
bM = np.ones((h,w))*0.1
wM = np.ones((w,h,h))
dB = np.zeros((h,w))
dW = np.zeros((w,h,h))
ii =0
#graph
g= []
for k in range(10):
    for i in range(len(inputVals)):
        A = inputVals[i][0]
        inputVals[i].pop(0)
        iv = np.asarray(inputVals[i]).reshape(1,-1)
        aM,zM = nu.Calculate(iv, wM, bM)
        db,dw,c= nu.BackPropogate(iv, aM, zM, bM, wM, A)
        g.append(c)
        dB+=db
        dW+=dW
        ii+=1
        if i%50 ==0 or i==len(inputVals)-1:
            print("Propogate")
            dB/=ii
            dW/=ii
            ii =0
            bM -= (dB*lr)
            wM -= (dW*lr)
            dB*=0
            dW*=0
        inputVals[i].insert(0,A)
        
plt.plot(g)
plt.show()