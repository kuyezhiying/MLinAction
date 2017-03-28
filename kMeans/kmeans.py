from numpy import mat, shape, sqrt, power, zeros, nonzero, random, inf, mean, sum, sin, cos, pi, arccos
import urllib
import json
from time import sleep
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines() :
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

'''
Compute euclidean distance between vectors
'''
def EuclideanDistance(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

'''
Randomly generate k centroids from the data set
'''
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n) :
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

'''
K-means algorithm, which may converge to a local minimum
'''
def kMeans(dataSet, k, distFunc = EuclideanDistance, initCentFunc = randCent):
    m = shape(dataSet)[0]
    clusterAssignment = mat(zeros((m, 2)))
    # initial centroids for k clusters
    centroids = initCentFunc(dataSet, k)
    clusterChanged = True
    while clusterChanged :
        clusterChanged = False
        # for each point, find the cluster with minimum distance
        for i in range(m) :
            minDist = inf; minIndex = -1
            # find the nearest centroid
            for j in range(k) :
                dist = distFunc(dataSet[i, :], centroids[j, :])
                if dist < minDist :
                    minDist = dist; minIndex = j
            if clusterAssignment[i, 0] != minIndex :
                clusterChanged = True
            clusterAssignment[i, :] = minIndex, minDist ** 2
        #print centroids
        # update the value of centroids
        for cluster in range(k) :
            ptsInCluster = dataSet[nonzero(clusterAssignment[:, 0].A == cluster)[0]]
            centroids[cluster, :] = mean(ptsInCluster, axis = 0)
    return centroids, clusterAssignment

'''
Bisecting k-means algorithm, which can converge to the global minimum
'''
def biKMeans(dataSet, k, distFunc = EuclideanDistance):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    # initial the cluster with all points
    centroid0 = mean(dataSet, axis = 0).tolist()[0]
    centList = [centroid0]
    for j in range(m) :
        clusterAssment[j, 1] = distFunc(dataSet[j, :], mat(centroid0)) ** 2
    while (len(centList) < k) :
        lowestSSE = inf
        # try to split each cluster and compute SSE, select the cluster with minimum SSE to split
        for i in range(len(centList)) :
            # split the ith cluster into two sub clusters
            ptsInCurClust = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAssment = kMeans(ptsInCurClust, 2, distFunc)
            # sum of squared error after splitting the cluster
            splitSSE = sum(splitClustAssment[:, 1])
            # sum of squared error of no divided clusters
            noSplitSSE = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print "split SSE and no split SSE : ", splitSSE, noSplitSSE
            if (splitSSE + noSplitSSE) < lowestSSE :
                bestClustToSplit = i
                bestNewCentroids = centroidMat
                bestClustAssment = splitClustAssment.copy()
                lowestSSE = splitSSE + noSplitSSE
        # update the cluster of divided points
        bestClustAssment[nonzero(bestClustAssment[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAssment[nonzero(bestClustAssment[:, 1].A == 0)[0], 0] = bestClustToSplit
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestClustToSplit)[0], :] = bestClustAssment
        print "the best cluster to split is : ", bestClustToSplit
        print "the length of the best cluster assignment is : ", len(bestClustAssment)
        # update centroids
        centList[bestClustToSplit] = bestNewCentroids[0, :].tolist()[0]
        centList.append(bestNewCentroids[1, :].tolist()[0])
    return mat(centList), clusterAssment

'''
Grab the latitude and longitude of a geographic location
'''
def geoGrab(streetAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (streetAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params
    print yahooApi
    c = urllib.urlopen(yahooApi)
    return json.loads(c.read())

def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines() :
        lineArr = line.strip().split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0 :
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print "%s\t%f\t%f" % (line, lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else :
            print "error in fetching location information."
        sleep(1)
    fw.close()

'''
Spherical distance
'''
def distSLC(vecA, vecB):
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0

def plotClubClusters(numClust = 5):
    dataList = []
    for line in open('places.txt').readlines() :
        lineArr = line.strip().split('\t')
        dataList.append([float(lineArr[4]), float(lineArr[3])])
    dataMat = mat(dataList)
    centroids, clustAssment = biKMeans(dataMat, numClust, distFunc = distSLC)
    print centroids
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '<', '>']
    axprops = dict(xticks = [], yticks = [])
    ax0 = fig.add_axes(rect, label = 'ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label = 'ax1', frameon = False)
    for i in range(numClust) :
        ptsInCurCluster = dataMat[nonzero(clustAssment[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurCluster[:, 0].flatten().A[0], ptsInCurCluster[:, 1].flatten().A[0], marker = markerStyle, s = 90)
    ax1.scatter(centroids[:, 0].flatten().A[0], centroids[:, 1].flatten().A[0], marker = '+', s = 300)
    plt.title('Clustering night entertainment clubs in Potland city')
    plt.show()


if __name__ == '__main__':
    '''
    dataMat = mat(loadDataSet('testSet2.txt'))
    
    centroids, clustAssment = kMeans(dataMat, 4)
    print centroids, clustAssment
    
    centList, clusterAssment = biKMeans(dataMat, 3)
    print centList, clusterAssment
    
    massPlaceFind('portlandClubs.txt')
    '''
    
    plotClubClusters(3)