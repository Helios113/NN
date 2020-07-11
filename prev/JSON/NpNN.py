import numpy as np

class layer():
    def __init__ (self,function,weights,biases,zArray,output,number,dq,dz,db,dw):
        self.function = function
        self.weights = weights
        self.biases = biases
        self.zArray = zArray
        self.output = output
        self.number = number
        self.dq = dq
        self.db = db
        self.dz = dz
        self.dw = dw    

    
def sigmoidD(x):
    #return 1
    if(x>0):
        return 1
    else:
        return 0
    #return 1-(np.tanh(x)**2)
    
    
    
def Construct_Layer(neurons,previousNeurons, function, number):
    l1 = np.ones((neurons,1),dtype = np.float)
    w = (np.random.rand(neurons,previousNeurons))
    m = max(a.all(w))
    mm = min(w)
    m = m-mm
    w = (w-mm)/m
    w-=0.5
    w*=2
    #b = (np.random.rand(neurons,1))/neurons
    b = np.zeros((neurons,1),dtype = np.float)
    z = np.zeros((neurons,1),dtype = np.float)
    dw = np.zeros((neurons,previousNeurons),dtype = np.float)
    db = np.zeros((neurons,1),dtype = np.float)
    dz = np.zeros((neurons,1),dtype = np.float)
    dq = np.zeros((neurons,1),dtype = np.float)
    ll = layer(function,w,b,z,l1,number,dq,dz,db,dw)
    return ll

def Construct_Input(vals):
    l1 = np.asarray(vals, dtype = np.float)
    ll = layer(0,0,0,[],l1,0,0,0,0,0)
    return ll
    
def ComputeLayer(l0,l1):
    a = np.matmul(l1.weights,l0.output).reshape(-1,1)
    a+=l1.biases
    l1.zArray = a
    if l1.output.shape == (1,1):
        l1.output = a
    else:
        l1.output = l1.function(a,0)
    return l1

def BackPropegate(l1,l0,y):
    if l1.output.shape == (1,1):
        l1.dq[0,0] = (l1.output[0]-y)
        l1.dz = 1
        r = np.multiply(l1.dq,l1.dz)
        l1.dw += np.matmul(r,l0.output.reshape(1,-1))
        l0.dq = np.matmul(l1.weights.T,r)
    else:
        r = np.multiply(l1.dq,l1.dz)
        l1.dw += np.matmul(r,l0.output.reshape(1,-1))
        if y == True:
            l0.dz = l0.function(l0.zArray,1).reshape(-1,1)
            l0.dq = np.matmul(l1.weights.T,r)
    l1.db += l1.dq*l1.dz
    return l1,l0
    
def Calculate(bM,wM,vals):
    rows,cols = bM.shape
    aM = np.zeros((rows,cols))
    zM = np.zeros((rows,cols))
    #fS = np.vectorize(sigmoid, otypes=[np.float])
    #First layer calcs
    aM[0,:] = np.asarray(vals)
    for i in range(1,rows):
        zM[i,:] = np.matmul(aM[i-1,:].reshape(1,-1),wM[i-1])[0,:]+bM[i-1,:]
        #print(fS(zM[i,:]))
        #aM[i,:] = fS(zM[i,:])
    
    aM[rows-1,0] = zM[rows-1,0]

    return aM,zM
    
def NegGrad(aM,zM,wM,ans):
    fS = np.vectorize(sigmoidD, otypes=[np.float])
    rows,cols = aM.shape;
    da = np.zeros((rows,cols),dtype = np.float)
    #dz = np.zeros((rows,cols),dtype = np.float)
    dw = np.zeros((rows-1,cols,cols),dtype = np.float)
    da[rows-1,0] = 2*(aM[rows-1,0]-ans)
    dz = fS(zM)
    dz[rows-1,0] = 1
    #--------Correct--------#
    #dz[rows-2,:] = sigd(a)
    for i in range(rows-2,-1,-1):
        dw[i] = np.matmul(aM[i,:].reshape(-1,1),(dz[i+1,:]*da[i+1,:]).reshape(1,-1))
        #-------correct--------#
        
        if(i>0):
            da[i,:] =np.matmul(wM[i], (dz[i+1,:]*da[i+1,:]).reshape(-1,1))[:,0]
        #-------correct--------#
    db = da*dz
    return (db,dw)
    


