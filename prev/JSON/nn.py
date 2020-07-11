import numpy as np
import random

class Neuron:
    def __init__(self, connectionsIn,w,b,a,z,d3):
        self.connectionsIn = connectionsIn
        self.w = w
        self.b = b
        self.a = a
        self.z = z
        self.d3 = d3 #dz/da[-1]
                
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoidD(x):
    return sigmoid(x)*(1-sigmoid(x))

def NegativeGrad(iiii):
    Nt,vals = iiii
    wih,h = Nt.shape
    #print(Nt.shape)
    t =[]
    q= 2*(Nt[wih-1][0].a-vals[0])
    print("Co",q)
    r=1
    vC = np.ndarray((wih,h),dtype = object)
    Nt[wih-1][0].d3 =1     
    for i in range(len(Nt[wih-1][0].w)):
        t.append(Nt[wih-2][Nt[wih-1][0].connectionsIn[i]].a*r*q)
        Nt[wih-2][Nt[wih-1][0].connectionsIn[i]].d3+= (Nt[wih-1][0].w[i]*r*q)
    
    vC[wih-1][0] = [t,q*r]
    for i in range(wih-2,-1,-1):
        for j in range(h):
            t =[]
            q = Nt[i][j].d3
            r = sigmoidD(Nt[i][j].z)
            for k in range(len(Nt[i][j].w)):
                if i > 0:
                    t.append(Nt[i-1][Nt[i][j].connectionsIn[k]].a*r*q)
                    Nt[i-1][Nt[i][j].connectionsIn[k]].d3+=(Nt[i][j].w[k]*r*q)
                else:
                    t.append(vals[Nt[i][j].connectionsIn[k]]*r*q)
            vC[i][j] = [t,q*r]
    return vC

def CalculateValue(iiii):
    Values = iiii[0]
    Nt = iiii[1]
    #print("has entered CAlc")
    #Values.insert(0, dT)
    wih,h = Nt.shape
    for j in range(h):
        val = 0
        for k in range(len(Nt[0][j].connectionsIn)):
            val+= Values[Nt[0][j].connectionsIn[k]]*Nt[0][j].w[k]
        val+=Nt[0][j].b
        Nt[0][j].z = val
        Nt[0][j].a = sigmoid(val)
    for i in range(1,wih-1):
        for j in range(h):
            val = 0
            for k in range(len(Nt[i][j].connectionsIn)):
                val+= Nt[i-1][Nt[i][j].connectionsIn[k]].a*Nt[i][j].w[k]
            val+=Nt[i][j].b
            Nt[i][j].z = val
            Nt[i][j].a = sigmoid(val)
    val =0
    for k in range(len(Nt[wih-1][0].connectionsIn)):
        val+= Nt[wih-2][Nt[wih-1][0].connectionsIn[k]].a*Nt[wih-1][0].w[k]
    val+=Nt[wih-1][0].b
    Nt[wih-1][0].z = val
    Nt[wih-1][0].a = val
    #print(val)
    return Nt

def Funk(valLen):
    print("has entered Funk")
    h = valLen
    wih = int(random.random()*2)+1
    h1 = int(random.random()*h)+1
    netwrk = np.ndarray((wih+1,h1), dtype = object)
    for j in range(h1):
        conList=[]
        numOfCon = int(random.random()*(h-1))+1
        NcA = 0
        for k in range(numOfCon):
            conNum = int(random.random()*(h-1))
            if conNum not in conList:
                NcA +=1
                conList.append(conNum)
        wght = [0.2] * NcA
        netwrk[0][j] = Neuron(conList,wght,0,0,0,0)
        
    
    
    print("has finished 1st for")
    for i in range(1,wih):
        for j in range(h1):
            conList=[]
            numOfCon = int((random.random()*h1))+1
            NcA = 0
            if numOfCon>h1:
                numOfCon-=1
            for k in range(numOfCon):
                conNum = int(random.random()*h1)
                if conNum == h1:
                    conNum-=1
                if conNum not in conList:
                    NcA +=1
                    conList.append(conNum)
            wght = [0.2] * NcA
            netwrk[i][j] = Neuron(conList,wght,0,0,0,0)
    ccc = list(range(h1))
    wght = [0.2] * len(ccc)
    netwrk[wih][0] = Neuron(ccc,wght,0,0,0,0)
    print("h1")
    return netwrk
    



def MatrixAddition(mat):
    wih,h = mat[0].shape
    for l in range(1,len(mat)):
        for i in range(wih-1):
            for j in range(h):
                for k in range(len(mat[0][i][j][0])):
                    mat[0][i][j][0][k]+=mat[l][i][j][0][k]
                mat[0][i][j][1]+=mat[l][i][j][1]
        for k in range(len(mat[0][wih-1][0][0])):
            mat[0][wih-1][0][0][k]+=mat[l][wih-1][0][0][k]
        mat[0][wih-1][0][1]+=mat[l][wih-1][0][1]     
    return mat[0]
def MatrixAvg(mat1,fac):
    wih,h = mat1.shape
    for i in range(wih-1):
        for j in range(h):
            for k in range(len(mat1[i][j][0])):
                mat1[i][j][0][k]/=fac
            mat1[i][j][1]/=fac
    for k in range(len(mat1[wih-1][0][0])):
        mat1[wih-1][0][0][k]/=fac
    mat1[wih-1][0][1]/=fac
    return mat1

def ChangeMatrix(Nt,mat1):
    wih,h = Nt.shape
    for i in range(wih-1):
        for j in range(h):
            for k in range(len(Nt[i][j].w)):
                Nt[i][j].w[k]+=mat1[i][j][0][k]
            Nt[i][j].b+=mat1[i][j][1]
    for k in range(len(Nt[wih-1][0].w)):
        Nt[wih-1][0].w[k]+=mat1[wih-1][0][0][k]
    Nt[wih-1][0].b+=mat1[wih-1][0][1]
    print("Change has been applied")
    print(Nt[wih-1][0].a)
    return Nt

