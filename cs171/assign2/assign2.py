
"""
@author: Kevin Lynn
861150305
"""
##q0 data
import matplotlib.pyplot as plt
import numpy as np
import math
import operator

'''
Headers = [ 'clump_thickness', 'cell_size','cell_shape', 'marginal adhesion', 'single_cell_size',
           'bare_nuclei', 'bland_chromation', 'normal_nucleoli', 'mitoses','class']
Data = np.genfromtxt('breast-cancer-wisconsin.data.txt', delimiter=',', dtype=float, missing_values='?',
                     filling_values = 0,names=Headers )
'''
fname = 'breast-cancer-wisconsin.data.txt'
Data = np.loadtxt(fname, delimiter=',') ##ignore/removed data points with missing values
Data = Data.reshape(683,11) ##used to be 699 data points
Data = Data[:,(1,2,3,4,5,6,7,8,9,10)]

#print(Data)
#print(len(Data))
sp = len(Data) * .8
sp = math.ceil(sp)

#print(sp)
traindata = Data[:sp]
testdata = Data[sp:]
x_train = testdata[:,9]
y_train = traindata[:,9]
##q1 k nearest neighbor
def distance(x,y,p):
    d = 0;
    for i in range(len(x)):
        d += (abs(x[i] - y[i]))**p
    return (d**(1/p))


def get_k_neighbors(trainingData, test, y_train, k, p):
    dis = []
    Cnt = 0
    for x in trainingData:
        newDistance = distance(x, test, p)
        dis.append((newDistance, y_train[Cnt]))
        Cnt += 1
    dis = sorted(dis, key=operator.itemgetter(0))
    
    kdis = [val[1] for val in dis[:k]]
    return kdis
    

def knn_classifier(x_test, x_train, y_train, k, p):
    y_pred = []
    cnt2 = 0
    cnt4 = 0
    for testValue in x_test:
        
        neighbors = get_k_neighbors(x_train, testValue, y_train, k, p)
        
        for i in range(k):
            if neighbors[i] == 2.0:
                cnt2 +=1
            else:
                cnt4 +=1
        #print (cnt2,cnt4)
        if cnt2 > cnt4:
            y_pred.append(2.0)
        elif cnt2 == cnt4:
            y_pred.append(neighbors[0])
        else:
            y_pred.append(4.0)
        cnt2 = 0
        cnt4 = 0
    
    return y_pred
##q2 cross validation
def crossval(data,k,p):
    np.random.shuffle(data)
    ##10 folds
    foldsize = int(np.size(data,0)/10)
    #print(foldsize)
    error = []
    acc= []
    sensitivity = []
    specificity = []
    start = 0
    end = foldsize
    for i in range(10):
        ##add remaining data (3 extra data points) 
        if i == 9:
            end = np.size(data,0)
            foldsize = end - start
            #print(foldsize)
        ##test and train data 
        x_test = data[start:end,(0,1,2,3,4,5,6,7,8)]
        x_train = data[:,(0,1,2,3,4,5,6,7,8)]
        y_train = data[:,9]
        next_test = np.arange(start,end)
        x_train = np.delete(x_train,next_test,0)
        y_train = np.delete(y_train,next_test,0)
        
        ##get data for comparison
        actual_data = data[start:end,9]
        
        ##knn
        ypred = knn_classifier(x_test,x_train,y_train,k,p)
        ##error accuracy sensitivity specificity 
        ##sensitivity = TP/P = TP/(TP + FN)
        ##specificity = TN/N = TN/(TN + FP)
        err=TP=P=TN=N = 0
        for val1,val2 in zip(actual_data,ypred):
            if val1 == 2:
                P += 1
            elif val1 == 4:
                N +=1
            if val1 == 2 and val2 == 2:
                TP +=1
            elif val1 == 4 and val2 == 4:
                TN +=1
            if val1 != val2:
                err +=1
        sensitivity.append(TP/P)
        specificity.append(TN/N)
        error.append(err/foldsize)
        acc.append(1- error[i])
        start = end
        end = end + foldsize
    return(error,acc,sensitivity,specificity)
'''
q1 accuracy
'''
result = knn_classifier(testdata,traindata,y_train,1,2)
#print(result)
sameCount = 0

for i,j in zip(x_train, result):
    if i == j:
        sameCount += 1

print(sameCount/len(x_train) * 100)
'''
q2 cross validation graphs
'''
xacc = [0]*10
xsens = [0]*10
xspec = [0]*10
xerror = [0]*10
a_std = []
a_mean = []
sens_std= []
sens_mean= []
spec_std= []
spec_mean = []
#crossval(Data,1,1)
p = 2
for i in range(10):
    print(i)
    xerror[i],xacc[i],xsens[i],xspec[i] = crossval(Data,i+1,p)
    a_std.append(np.std(np.array(xacc[i])))
    a_mean.append(np.mean(np.array(xacc[i])))
    sens_std.append(np.std(np.array(xsens[i])))
    sens_mean.append(np.mean(np.array(xsens[i])))
    spec_std.append(np.std(np.array(xspec[i])))
    spec_mean.append(np.mean(np.array(xspec[i])))

a = plt.figure(1)
x = np.arange(1,11)
y = a_mean
a = plt.errorbar(x, y, xerr=0, yerr=a_std,color='green',ecolor='crimson', capsize=5, capthick=2)
plt.xlabel("K number of neighbors")
plt.ylabel("Accuracy")
plt.title("Accuracy with p = " + str(p))

b= plt.figure(2)
x = np.arange(1,11)
y = sens_mean
b = plt.errorbar(x, y, xerr=0, yerr=sens_std,color='green',ecolor='crimson', capsize=5, capthick=2)
plt.xlabel("K number of neighbors")
plt.ylabel("sensitivity")
plt.title("Sensitivity with p = " + str(p))

c=plt.figure(3)
x = np.arange(1,11)
y = spec_mean
c = plt.errorbar(x, y, xerr=0, yerr=spec_std,color='green',ecolor='crimson', capsize=5, capthick=2)
plt.xlabel("K number of neighbors")
plt.ylabel("specificity")
plt.title("Specificity with p = " + str(p))
plt.show()