from numpy import mat, shape, zeros, ones, exp, log, inf, multiply, sign, array

def loadDataPoints():
    dataArr = [[1. , 2.1],
               [2. , 1.1],
               [1.3, 1. ],
               [1. , 1. ],
               [2. , 1. ]]
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataArr, classLabels

def loadDataSet(fileName):
    numFeatures = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines() :
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeatures) :
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

'''
Classify with decision stump by compare with threshold value
'''
def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    returnArr = ones((shape(dataMat)[0], 1))
    if threshIneq == 'lt' :
        returnArr[dataMat[:, dimen] <= threshVal] = -1.0
    else :
        returnArr[dataMat[:, dimen] > threshVal] = -1.0
    return returnArr

'''
Build best decision stump from data
'''
def buildStump(dataArr, classLabels, weights):
    dataMat = mat(dataArr); labelMat = mat(classLabels).T
    m, n = shape(dataMat)
    numSteps = 10.0; bestStump = {}; bestClassEst = mat(zeros((m, 1)))
    minError = inf
    # for each feature
    for i in range(n) :
        rangeMin = dataMat[:, i].min(); rangeMax = dataMat[:, i].max(); stepSize = (rangeMax - rangeMin) / numSteps
        # for each step to set a threshold value (besides the range(rangeMin, rangeMax) is also reasonable)
        for j in range(-1, int(numSteps) + 1) :
            # for each threshold inequality (less than or greater than)
            for inequal in ['lt', 'gt'] :
                threshVal = rangeMin  + float(j) * stepSize
                predicateVals = stumpClassify(dataMat, i, threshVal, inequal)
                # compute weighted error for data
                errorArr = mat(ones((m, 1)))
                errorArr[predicateVals == labelMat] = 0
                weightedError = weights.T * errorArr
                #print "split : dimension [%d], threshold value [%.2f], threshold inequality [%s], weighted error [%.3f]" % \
                #      (i, threshVal, inequal, weightedError)
                if weightedError < minError :
                    minError = weightedError
                    bestClassEst = predicateVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['inequal'] = inequal
    return bestStump, bestClassEst, minError

'''
Train AdaBoost on decision stump
'''
def trainAdaBoostOnDS(dataArr, classLabels, numIter = 40):
    weakClassifiers = []
    m = shape(dataArr)[0]
    weights = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIter) :
        bestStump, classEst, error = buildStump(dataArr, classLabels, weights)
        #print "weights : ", weights.T
        # compute weight for the weak decision stump classifier [alpha = 0.5 * log((1 - error) / error)]
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassifiers.append(bestStump)
        #print "class estimation : ", classEst.T
        # update sample weights [Wi(t+1) = Wi(t)*exp(+/-alpha) / sum(Wi)]
        expon = exp(multiply(-1 * alpha * mat(classLabels).T, classEst))
        weights = multiply(weights, expon)
        weights = weights / weights.sum()
        # compute aggregated class estimation
        aggClassEst += alpha * classEst
        #print "aggregated class estimation : ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        #print "total error : ", errorRate
        if errorRate == 0.0 :
            print "Finished training after [%d] iterations." % (i + 1)
            break;
    return weakClassifiers, aggClassEst

def classify(testData, classifierArr):
    dataMat = mat(testData)
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)) :
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['inequal'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        #print aggClassEst
    return sign(aggClassEst)

def testColic():
    dataArr, labelArr = loadDataSet('horseColicTraining.txt')
    classifierArr, aggClassEst = trainAdaBoostOnDS(dataArr, labelArr, 50)   # 1, 10, 50, 100, 500, 1000, 10000
    plotROC(aggClassEst, labelArr)
    
    testDataArr, testLabelArr = loadDataSet('horseColicTest.txt')
    numTestCases = shape(testDataArr)[0]
    prediction = classify(testDataArr, classifierArr)
    
    errorArr = mat(ones((numTestCases, 1)))
    errorCount = errorArr[prediction != mat(testLabelArr).T].sum()
    errorRate = errorCount / numTestCases
    print "[%d] error cases in prediction on [%d] samples, the test error rate is : [%.2f]" % (errorCount, numTestCases, errorRate)

def plotROC(predicateStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum = 0.0
    numPositiveClasses = sum(array(classLabels) == 1.0)
    yStep = 1.0 / float(numPositiveClasses)
    xStep = 1.0 / float(len(classLabels) - numPositiveClasses)
    sortedIndicies = predicateStrengths.argsort(axis = 0)
    indexList = array(sortedIndicies).flatten()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in indexList :
        if classLabels[index] == 1.0 :
            deltaX = 0; deltaY = yStep
        else :
            deltaX = xStep; deltaY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - deltaX], [cur[1], cur[1] - deltaY], c = 'b')
        cur = (cur[0] - deltaX, cur[1] - deltaY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print "the AUC (Area Under the Curve) is : ", ySum * xStep


#if __name__ == '__main__':
def main():
    '''
    dataArr, classLabels = loadDataPoints()
    weights = mat(ones((5, 1)) / 5)
    bestStump, bestClassEst, minError = buildStump(dataArr, classLabels, weights)
    print bestStump, bestClassEst, minError
    
    weakClassifiers, aggClassEst = trainAdaBoostOnDS(dataArr, classLabels, 9)
    print weakClassifiers, aggClassEst
    
    result = classify([[5, 5], [0, 0]], weakClassifiers)
    print result
    '''
    testColic()