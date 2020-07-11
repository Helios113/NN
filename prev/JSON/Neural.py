import numpy as np
def sig(x):
    return np.tanh(x)
def dsig(x):
    return 1-(np.tanh(x)**2)

def Calculate(inputVals, wM,bM):
    fS = np.vectorize(sig, otypes=[np.float])
    size = bM.shape
    w = size[1]
    h = size[0]
    aM = np.zeros((h,w),dtype = float)
    zM = np.zeros((h,w),dtype = float)
    #First layer calcs
    for i in range(w):
        a = np.matmul(inputVals,wM[i])
        a+=bM[:,i].reshape(1,-1) 
        zM[:,i] = a.reshape(-1,1)[:,0]
        if i<w-1:
            aM[:,i] = fS(zM[:,i])
        else:
            aM[0,i] = zM[0,i]
    return aM,zM

def BackPropogate(inputVals,aM,zM,bM,wM,A):
    fdS = np.vectorize(dsig, otypes=[np.float])
    size = bM.shape
    w = size[1]
    h = size[0]
    dw = np.zeros((w,h,h),dtype = float)
    q = np.zeros((h,w),dtype = float)
    r = fdS(zM)
    q[0,w-1] = 2*(aM[0,w-1]-A)
    r[0,w-1] = 1
    
    for i in range(w-1,-1,-1):
        if i == w-1:
            dw[i,:,0] = aM[:,i-1]*q[0,i] 
        elif i>0:
            dw[i] = np.matmul(aM[:,i-1].reshape(1,-1),(q[:,i]*r[:,i]).reshape(-1,1))
        else:
            dw[i] = np.matmul(inputVals.reshape(1,-1),(q[:,i]*r[:,i]).reshape(-1,1))
        q[:,i-1] = np.matmul((q[:,i]*r[:,i]).reshape(1,-1),wM[i].T).reshape(-1,1)[:,0]
    db = q*r
    return db,dw,(aM[0,w-1]-A)