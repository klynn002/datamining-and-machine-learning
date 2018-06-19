
"""

@author: Kevin Lynn
861150305
"""

import numpy as np
import matplotlib.pyplot as plt


"""
Q0 data
"""
fname = 'iris_data.txt'
data = np.loadtxt(fname, delimiter = ',',usecols = (0,1,2,3) )
data = data.reshape(150,4)

"""
Q1 K means clustering
"""

def distance(x,y,p):
    distance = 0;
    for i in range(len(x)):
        distance += (abs(x[i] - y[i]))**p
    return ((distance)**(1/p))
def make_centroids(x_input,K):
    cluster_k = np.random.randint(0,len(x_input)-1,K)
    init_centroids = np.array([])
    for i in cluster_k:
        #print(i)
        init_centroids = np.append(init_centroids,data[i],axis= 0)
    init_centroids = init_centroids.reshape(K,4)
    return init_centroids
def k_means_cs171(x_input, k , init_centroids):
    old = np.zeros(150)
    new = np.arange(150)
    #compares clusters and repeats if there are still changes
    while(not(np.array_equal(old, new))):
        old = np.array(new, copy= True)
        for i in range(np.size(x_input,0)):
            closest_cent = 99999.9
            #assign points to closest centroid
            for j in range(k):
                #p = 2 for euclidean 
                d = distance(x_input[i], init_centroids[j],2)
                if(d < closest_cent):
                    closest_cent = d
                    new[i] = j
                    
        #new centroids as mean of all points in cluster
        centroid = np.zeros((k,4))
        cnt = np.zeros(k)
        for x in range(np.size(x_input,0)):
            for y in range(k):
                if(new[x] == y):
                    centroid[y] = centroid[y] + x_input[x]
                    cnt[y] += 1
        for n in range(k):
            init_centroids[n] = np.true_divide(centroid[n], cnt[n])
    return new, init_centroids

def sumofsquare_error(centroids, assignments, data):
    total = 0
    for i in range(np.size(assignments)):
        for j in range(np.size(centroids,0)):
            if(assignments[i] == j):        
                #sum up the distances from all points squared
                total += distance(data[i],centroids[j],2)**2  
                
    return total

#sum of squares error for k = 3 
part1 = 0
initial_centroids = make_centroids(data,3)     
cluster_assignments, cluster_centroids = k_means_cs171(data,3,initial_centroids)

#print(cluster_centroids)
part1 = sumofsquare_error(cluster_centroids, cluster_assignments, data)
print(part1)
"""
Q2 knee plot/ sensitivity analysis
"""
def Kneeplot(data):
    klist =list(range(1,11))
    kneelist = []
    
    for entry in klist:
        k_init_centroids = make_centroids(data,entry)
        kassignments, kcentroids = k_means_cs171(data,entry,k_init_centroids)
        error = sumofsquare_error(kcentroids, kassignments, data)
        kneelist.append((entry,error))
    return kneelist
#first knee plot no sensitivity analysis
firstplot = Kneeplot(data)
xaxis = [x[0] for x in firstplot]
yaxis = [y[1] for y in firstplot]
plt.figure(1)
plt.xlabel("K number of clusters")
plt.ylabel("sum of square error")
plt.xticks(list(range(11)))
plt.plot(xaxis,yaxis,'-o')
plt.show()

def sensitivity_analysis(data,max_iter):
    senslist = []
    for n in range(max_iter):
        kneeplots = Kneeplot(data)
        plotlist = [entry[1] for entry in kneeplots]
        senslist.append(plotlist)
    means = np.mean(senslist,axis = 0)
    stdev = np.std(senslist,axis = 0)
    return means, stdev
 #knee plots with sensitivity analysis         
iter_list = [2,10,100]
figcnt = 1
xlist = list(range(1,11))
for max_iter in iter_list:
    kmean, kstdev = sensitivity_analysis(data, max_iter)
    figcnt += 1
    plt.figure(figcnt)
    plt.xlabel("K number of clusters")
    plt.ylabel("sum of square error")
    plt.xticks(list(range(11))) 
    plt.errorbar(xlist, kmean, yerr = kstdev, ecolor = 'crimson')
    plt.show()
            

            