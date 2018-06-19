import matplotlib.pyplot as plt
import numpy as np

#Q0
#names and data
irisHeaders = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
wineHeaders = ['alcohol', 'malic_acid', 'ash']
irisData = np.genfromtxt('iris.data.txt', delimiter=',', dtype=None, names=irisHeaders)
wineData = np.genfromtxt('wine.data.txt', delimiter=',', dtype=None, names=wineHeaders,usecols = (1,2,3))
#classes
setosa = irisData[0:50]
versicolor = irisData[50:100]
virginica = irisData[100:150]
wineclass1 = wineData[0:59]
wineclass2 = wineData[59:130]
wineclass3 = wineData[130:178]


#Q1 histogram

def histogram(data, binSize, figure, subIndex, title):
    Min = np.min(data)
    Max = np.max(data)
    width = (Max - Min)/binSize
    
    # Index to keep track of bar data
    lowerBound = Min
    bars = []
    indicies = []
    
    
    for bar in range(binSize):
        if bar != binSize-1:
            # find frequency of given range
            freq = ((lowerBound <= data) & (data < lowerBound+width)).sum()
            bars.append(freq)
            
        else:
            freq = ((lowerBound < data) & (data <= lowerBound+width)).sum()
            bars.append(freq)
        indicies.append(lowerBound)
        lowerBound+=width
        
    plt.figure(figure)
    plt.subplot(2,2,subIndex)
    plt.bar(indicies, bars, width, align='edge')
    plt.title(title)
    plt.xticks(np.arange(Min, Max,width))
    plt.xticks(rotation =90)
#rename things in function to get desired graph and change binsize
histogram(virginica['sepal_length'], 10, 1, 1,"Virginica Sepal Length")
histogram(virginica['sepal_width'], 10, 1, 2, "Virginica Sepal Width")
histogram(virginica['petal_length'], 10, 1, 3, "Virginica Petal Length")
histogram(virginica['petal_width'], 10, 1, 4, "Virginica Petal Width")

#histogram(wineclass3['alcohol'], 100, 1, 1,"class3 alcohol")
#histogram(wineclass3['malic_acid'], 100, 2, 1,"class3 malic acid")
#histogram(wineclass3['ash'], 100, 3, 1,"class3 ash")

plt.tight_layout()
plt.show()





#change names to get desired plot
plt.figure(2)
plt.boxplot(wineclass3['alcohol'],vert = False)
#plt.figure(3)
#plt.boxplot(wineclass3['malic_acid'],vert = False)
#plt.figure(4)
#plt.boxplot(wineclass3['ash'],vert = False)

#plt.show()

#Q2 
#pearson covariance/ stdev x * stdev y
def stdev(a):
    amean = np.mean(a)
    sum = 0
    for i in range(len(a)):
        sum += ((a[i]-amean)**2)
    return sum **.5
def correlation(x,y):

    xmean = np.mean(x)
    ymean = np.mean(y)
    
    num = 0.0

    for i in range(len(x)):
        num += (x[i]-xmean)*(y[i]-ymean)
    stdevx = stdev(x)
    stdevy = stdev(y)
    return( num /(stdevx*stdevy))

#correlation matrix
def corrMatrix(Data):    
    cmat = []
    for name1 in Data.dtype.names:
        if name1 == 'label':
            continue
        newRow = []
        for name2 in Data.dtype.names:
            if name2 != 'label' and name1 != 'label':
                corr = correlation(Data[name1], Data[name2])
                newRow.append(corr)
        cmat.append(newRow)
    return np.array(cmat)
irismat = corrMatrix(irisData)
plt.figure(3)
plt.imshow(irismat, cmap ='Blues')
plt.colorbar()
plt.show()
wineData2 = np.genfromtxt('wine.data.txt', delimiter=',', dtype=None)
winemat = corrMatrix(wineData2)
#plt.figure(3)
#plt.imshow(winemat, cmap ='Reds')
#plt.colorbar()
#plt.show()
#scatter plots
#changed names inside funtion to get desired plot
#sepal_width sepal_length petal_width petal_length
plt.figure(5)
plt.scatter(setosa['petal_width'], setosa['petal_width'], c='red')
plt.scatter(versicolor['petal_width'], versicolor['petal_width'], c='blue')
plt.scatter(virginica['petal_width'], virginica['petal_width'], c='green')
plt.show()

def distance(x,y,p):
    distance = 0
    for i in range(len(x)):
        if type(x[i]) is not np.float64 or type(y[i]) is not np.float64:
            continue
        else:
            distance = distance + (x[i] - y[i])**2
    if(p==1):
        return distance
    if(p == 2 ):
        return (distance **.5)
def lpmatrix(data,p):
    lpmat = []
    for row1 in data:
        newrow = []
        for row2 in data:
            newrow2 = distance(row1, row2, p)
            newrow.append(newrow2)
        lpmat.append(newrow)
    return np.array(lpmat)

irismat = lpmatrix(wineData,1)
plt.figure(4)
plt.imshow(irismat, cmap = 'Greens')
plt.colorbar()
plt.show()




