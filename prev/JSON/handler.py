import NpNN as nnn
import numpy as np
import comm as cm
import matplotlib.pyplot as plt
#sample = cm.getPrices("day")
#sample = cm.getCSVdata()
sample = []
nP = 30
data = cm.getStandardData(sample, nP, 2, 1/2,1/12,False)
cols = len(data[0])-1
plotting = []
plotting1 = []


def relu(x,y):
    if y==0:
        if(x>0):
            return x
        else:
            return 0
    else:
        if(x>0):
            return 1
        else:
            return 0
func = np.vectorize(relu)

layers = 5


NN = np.ndarray(layers, dtype = object)
NN[1] = nnn.Construct_Layer(30,cols,func,1)
NN[2] = nnn.Construct_Layer(20,30,func,2)
NN[3] = nnn.Construct_Layer(10,20,func,3)
NN[4] = nnn.Construct_Layer(1,10,func,4)
    

def Forward(vals):
    global NN
    NN[0] = nnn.Construct_Input(vals)
    for i in range(layers-1):
        NN[i+1] = nnn.ComputeLayer(NN[i],NN[i+1])
        
def BackProp(ans):
    global NN
    for i in range(layers-1,0,-1):
        if i ==layers-1:
            NN[i],NN[i-1] = nnn.BackPropegate(NN[i],NN[i-1],ans)
        elif i==1:
            NN[i],NN[i-1] = nnn.BackPropegate(NN[i],NN[i-1],False)
        else:
            NN[i],NN[i-1] = nnn.BackPropegate(NN[i],NN[i-1],True)
    
def RMS(k,lr):
    global NN
    for i in range(layers):
        print("layer", i)
        print(NN[i].weights)
        NN[i].dw/=k
        NN[i].weights-=(NN[i].dw*lr)
        NN[i].dw *=0
        NN[i].db/=k
        NN[i].biases-=(NN[i].db*lr)
        NN[i].db *=0
        
"""bMatrix = np.random.rand(h,w+1)
wMatrix = np.random.rand(d,h,h)
wCMatrix1 = np.random.rand(VectorSize,h)
wCMatrix2 = np.random.rand(h,1)
bMatrix -=0.5
wMatrix -=0.5
wCMatrix1 -=0.5
wCMatrix2 -=0.5

bMatrix *=0.1
"""
cnt=0
baseLr = 1
#Dynamic learning rate
#wMatrix-=0.5


dynamicLr = 0

plotting2 =[]

#dynamicLr = aMatrix[rows-1,0]-Ans

for i in range(1):
    for k in range(len(data)):
        Ans = data[k].pop(0)
        Forward(data[k])
        plotting1.append((NN[4].output[0,0]))
        plotting2.append(Ans)
        cnt+=1
        BackProp(Ans)
        if cnt == 30 or k == len(data)-1:
            plt.scatter(len(plotting1)-1,1)
            RMS(cnt,1)
            cnt =0
        data[k].insert(0,Ans)
        #print((i*5)+((k+1)/len(data)*5),"%")
plt.plot(plotting)
plt.plot(plotting1)
plt.plot(plotting2)
plt.show()