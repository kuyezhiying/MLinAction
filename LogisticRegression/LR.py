from numpy import exp, mat, shape, ones, array, arange
import random
import matplotlib.pyplot as plt

def loadDataSet():
    dataSet = []; classLabels = []
    fr = open('testSet.txt')
    for line in fr.readlines() :
        lineArr = line.strip().split()
        dataSet.append([1.0, float(lineArr[0]), float(lineArr[1])])
        classLabels.append(int(lineArr[2]))
    return dataSet, classLabels

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

'''
Optimal algorithm -- gradient ascent
'''
def gradientAscent(dataSet, classLabels):
    dataMatrix = mat(dataSet)
    labelMatrix = mat(classLabels).transpose()
    n = shape(dataMatrix)[1]
    # initialize weights matrix
    weights = ones((n, 1))
    alpha = 0.001       # step length
    maxCycle = 500      # iteration times
    for k in range(maxCycle) :
        kthPrediction = sigmoid(dataMatrix * weights)
        error = (labelMatrix - kthPrediction)
        # update weights with the gradient of whole data set
        weights = weights + alpha * dataMatrix.transpose() * error
        if k == maxCycle :
            print "Finding the weights fit the data set best. ", weights
    return weights

'''
Optimal algorithm -- stochastic gradient ascent
'''
def stocGradAscent(dataArr, classLabels):
    m, n = shape(dataArr)
    alpha = 0.01       # step length
    weights = ones(n)
    for i in range(m) :
        ithPrediction = sigmoid(dataArr[i] * weights)
        error = (classLabels[i] - ithPrediction)
        # update weights with the gradient of ith sample
        weights = weights + alpha * error * dataArr[i]
    return weights

'''
Optimal algorithm (online) -- advanced stochastic gradient ascent
'''
def advancedStocGradAscent(dataArr, classLabels, numIter = 1):
    m, n = shape(dataArr)
    weights = ones(n)
    for j in range(numIter) :
        dataIndex = range(m)
        for i in range(m) :
            # dynamically adjust step length for each iteration, to relieve the frequent data fluctuation
            alpha = 4 / (1.0 + j + i) + 0.01
            # randomly select a sample, to reduce the cyclic data fluctuation
            randIndex = int(random.uniform(0, len(dataIndex)))
            ithPrediction = sigmoid(dataArr[randIndex] * weights)
            error = (classLabels[randIndex] - ithPrediction)
            # update weights with the gradient of a randomly selected sample
            weights = weights + alpha * error * dataArr[randIndex]
            del(dataIndex[randIndex])
    return weights

'''
Plot the best fitting curve (decision boundary)
'''
def plotFittingLine(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n) :
        if int(labelMat[i]) == 1 :
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else :
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    # create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # plot data points
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    
    # plot fitting line
    # set w0*x0 + w1*x1 + w2*x2 = 0, x0 = 1  ==>>  x2 = -(w0 + w1*x1) / w2
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5 :
        return 1.0
    else :
        return 0.0

def colicTest(k):
    # collect training data
    frTrain = open('horseColicTraining.txt')
    trainingSet = []; trainingClasses = []
    for line in frTrain.readlines() :
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(21) :
            lineArr.append(float(curLine[i]))
        trainingSet.append(lineArr)
        trainingClasses.append(float(curLine[21]))
    
    # train LR model    
    #trainWeights = gradientAscent(trainingSet, trainingClasses)
    trainWeights = stocGradAscent(array(trainingSet), trainingClasses)
    #trainWeights = advancedStocGradAscent(array(trainingSet), trainingClasses, 500)
    
    # predicate and count number of error cases
    errorCount = 0; numTestVecs = 0.0
    frTest = open('horseColicTest.txt')
    for line in frTest.readlines() :
        numTestVecs += 1.0
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(21) :
            lineArr.append(float(curLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(curLine[21]) :
            errorCount += 1
    
    # compute error rate
    errorRate = float(errorCount) / numTestVecs
    print "the error rate of the %d iteration test is : %f" % (k, errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests) :
        errorSum += colicTest(k)
    print "after %d iterations, the average error rate is : %f" % (numTests, errorSum / float(numTests))


#if __name__ == '__main__':
def main():
    '''
    dataSet, classLabels = loadDataSet()
    
    weights = gradientAscent(dataSet, classLabels)
    print weights
    plotFittingLine(weights.getA())
    
    weights = stocGradAscent(array(dataSet), classLabels)
    print weights
    plotFittingLine(weights)
    
    weights = advancedStocGradAscent(array(dataSet), classLabels)
    print weights
    plotFittingLine(weights)
    '''
    multiTest()