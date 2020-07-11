import nn
import testies
nns = []
Pdata = testies.getPrices('day')
sampleNum = 500
sampleSize = 100
from multiprocessing import Pool
from multiprocessing import Process, Lock
#inputVals,tmf,ppr,mpr = testies.getStandardData(Pdata,sampleNum)
#testies.displayStdData(inputVals)
#inputVals.insert(0,0.5)


trainingData = testies.getTrainingData(1, 0.5, 1/60,sampleNum)

lln = sampleNum*2+1
print(len(trainingData))
nns.append(nn.Funk(lln))



cnt =0
ob =[]
tdl = len(trainingData)
ll = 0.002
coAvg =0
coPrev = 0
for j in range(len(trainingData)):
    if __name__ == '__main__':
        p = Pool()
        multiData = [trainingData[j],nns[0]]
        r = p.map_async(nn.CalculateValue,multiData)
        nns[0],co=r
        coAvg+=co
        ob1 = nn.NegativeGrad(nns[0],trainingData[j])
        if ob ==[]:
            ob = ob1
        else:
            ob = nn.MatrixAddition(ob,ob1)
        
       
        if cnt==99 or j ==tdl-1:
            coAvg/=cnt
            print("Adapt")
            negGrad = nn.MatrixAvg(ob, -(cnt+1)/ll)
            print(coAvg)
            nns[0] = nn.ChangeMatrix(nns[0],negGrad)
            ob =[]
            cnt =-1
            if coPrev !=0:
                if coPrev*coAvg <0:
                    ll/=2
            coPrev = coAvg
        cnt+=1
