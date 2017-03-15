from numpy import mat, linalg, corrcoef, mean, var, shape, eye, exp, zeros, inf, array, nonzero, multiply
import matplotlib.pyplot as plt
import random

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
Standard linear regression (SLR) model [w = (xT * x)^-1 * xT * y]
'''
def standardRegression(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # use library linalg (for linear algebra) to detect whether a matrix is singular
    if linalg.det(xTx) == 0.0 :
        print "xTx is a singular matrix, cannot do inverse."
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def plotSLR(xArr, yArr):
    # plot data points
    xMat = mat(xArr); yMat = mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    #plot the fitting line
    xCopy = xMat.copy()
    xCopy.sort(0)
    ws = standardRegression(xArr, yArr)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.title('Standard Linear Regression for points')
    plt.show()

'''
Locally Weighted Linear Regression (LWLR) model [w = (xT * W * x)^-1 * xT * W * y]
'''
def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    # compute sample weights with Gauss kernel [w(i, i) = exp(|xi - x| / (-2 * k^2))]
    weights = mat(eye((m)))
    for i in range(m) :
        diffMat = testPoint - xMat[i, :]
        weights[i, i] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTWx = xMat.T * weights * xMat
    # use library linalg (for linear algebra) to detect whether a matrix is singular
    if linalg.det(xTWx) == 0.0 :
        print "xTWx is a singular matrix, cannot do inverse."
        return
    ws = xTWx.I * (xMat.T * weights * yMat)
    return testPoint * ws

'''
Predict output y with LWLR model
'''
def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m) :
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def plotLWLR(xArr, yArr, k = 1.0):
    xMat = mat(xArr); yMat = mat(yArr)
    sortIndex = xMat[:, 1].argsort(0)
    xSort = xMat[sortIndex][:, 0, :]
    yHat = lwlrTest(xSort, xArr, yArr, k)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plot data points
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0], s = 2, c = 'red')
    #plot the fitting line
    ax.plot(xSort[:, 1], yHat)
    
    plt.title('Locally Weighted Linear Regression for points (k = %.3f)' % k)
    plt.show()

'''
Square error for regression models
'''
def squareError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()

def testAbalone():
    abX, abY = loadDataSet('abalone.txt')
    k = 10     # k = 0.1, 1, 10
    yHat = lwlrTest(abX[0 : 99], abX[0 : 99], abY[0 : 99], k)
    trainingError = squareError(abY[0 : 99], yHat.T)
    print "LWLR training error : ", trainingError
    yHatTest = lwlrTest(abX[100 : 199], abX[0 : 99], abY[0 : 99], k)
    testError = squareError(abY[100 : 199], yHatTest.T)
    print "LWLR test error : ", testError
    
    ws = standardRegression(abX[0 : 99], abY[0 : 99])
    yHatSLR = mat(abX[0 : 99]) * ws
    trainingErrorSLR = squareError(abY[0 : 99], yHatSLR.T.A)
    print "SLR training error : ", trainingErrorSLR
    yHatTestSLR = mat(abX[100 : 199]) * ws
    testErrorSLR = squareError(abY[100 : 199], yHatTestSLR.T.A)
    print "SLR test error : ", testErrorSLR
    
'''
Normalize the data
'''
def normalize(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xMeans = mean(xMat, 0); xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    return xMat, yMat
    
'''
Shrinkage method -- Ridge Regression (RR) [w = (xT * x + lambda * I)^-1 * xT * y]
'''
def ridgeRegression(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + lam * eye(shape(xMat)[1])
    # use library linalg (for linear algebra) to detect whether a matrix is singular
    # if the number of features are more than the number of samples and lambda = 0, the matrix denom would be singular
    if linalg.det(denom) == 0.0 :
        print "This matrix (xT * x + lambda * I) is singular, cannot do inverse."
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

'''
Compute ridge regression weights for different settings of lambda value
'''
def ridgeTest(xArr, yArr):
    xMat, yMat = normalize(xArr, yArr)
    # compute ridge regression weights
    numTestLambdas = 30
    wMat = zeros((numTestLambdas, shape(xMat)[1]))
    for i in range(numTestLambdas) :
        ws = ridgeRegression(xMat, yMat, exp(i-10))
        wMat[i, :] = ws.T
    return wMat

def plotRR():
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.xlabel('log(lambda)')
    plt.ylabel('ridge weights')
    plt.title('Ridge regression weights over different lambda settings')
    plt.show()

'''
Shrinkage method -- Forward Stepwise Linear Regression (FSLR)
'''
def forwardStepwiseLR(xArr, yArr, stepLen = 0.01, numIt = 100):
    xMat, yMat = normalize(xArr, yArr)
    n = shape(xMat)[1]
    weightsMat = zeros((numIt, n))
    weights = zeros((n, 1)); wsTest = weights.copy(); wsBest = weights.copy()
    for i in range(numIt) :
        #print weights.T
        minError = inf
        for j in range(n) :
            for sign in [-1, 1] :
                wsTest = weights.copy()
                wsTest[j] += sign * stepLen
                yTest = xMat * wsTest
                error = squareError(yTest.A, yMat.A)
                if error < minError :
                    minError = error
                    wsBest = wsTest
        weights = wsBest.copy()
        weightsMat[i, :] = weights.T
    return weightsMat
        
def plotFSLR(stepLen, numIt):
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = forwardStepwiseLR(abX, abY, stepLen, numIt)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.xlabel('times of iteration')
    plt.ylabel('FSLR weights')
    plt.title('FSLR weights over %d iterations (step = %.3f)' % (numIt, stepLen))
    plt.show()        

'''
Test ridge regression with cross validation
'''
def crossValidation(xArr, yArr, numFold = 10, numLams = 30):
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numFold, numLams))
    for i in range(numFold) :
        trainX = []; trainY = []
        testX = []; testY = []
        random.shuffle(indexList)
        # split training set and test set
        for j in range(m) :
            if j < m * 0.9 :
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else :
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
            weightsMat = ridgeTest(trainX, trainY)
        print weightsMat
        for k in range(numLams) :
            testXMat = mat(testX); trainXMat = mat(trainX)
            trainMatMean = mean(trainXMat, 0)
            trainMatVar = var(trainXMat, 0)
            testXMat = (testXMat - trainMatMean) / trainMatVar
            yHat = testXMat * mat(weightsMat[k, :]).T + mean(trainY)
            errorMat[i, k] = squareError(array(testY), yHat.T.A)
    print weightsMat
    print errorMat
    meanErrors = mean(errorMat, 0)
    minErrorMean = float(min(meanErrors))
    bestWeights = weightsMat[nonzero(meanErrors == minErrorMean)]
    print bestWeights
    xMat = mat(xArr); yMat = mat(yArr).T
    meanX = mean(xMat, 0); varX = var(xMat, 0)
    unReg = bestWeights / varX
    print "the best model from Ridge Regression is : \n", unReg
    print "with constant term : ", -1 * sum(multiply(meanX, unReg)) + mean(yMat)

def main():
    
    xArr, yArr = loadDataSet('ex0.txt')
    
    ws = standardRegression(xArr, yArr)
    print ws
    
    plotSLR(xArr, yArr)
    
    xMat = mat(xArr); yMat = mat(yArr)
    yHat = xMat * ws
    print corrcoef(yHat.T, yMat)
    
    plotLWLR(xArr, yArr, 0.01)
    
    testAbalone()
    
    plotRR()
    
    plotFSLR(0.005, 1000)
    
    abX, abY = loadDataSet('abalone.txt')
    crossValidation(abX, abY)
    
if __name__ == '__main__':
    main()
