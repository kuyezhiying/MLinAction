'''
CART : Classification And Regression Trees
'''

from numpy import mat, shape, ones, zeros, nonzero, mean, var, power, inf, linalg, corrcoef

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines() :
        curLine = line.strip().split('\t')
        floatLine = map(float, curLine)
        dataMat.append(floatLine)
    return dataMat

'''
Binary split of the data set
'''
def binarySplitDS(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

'''
Generate leaf node of regression tree, the node value is mean of the target variable
'''
def regLeaf(dataSet):
    return mean(dataSet[:, -1])

'''
Error estimation of regression tree, the error value is the total variance of the target variable
'''
def regError(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]

'''
Choose the best feature and value to split the data set
'''
def chooseBestSplit(dataSet, leafFunc = regLeaf, errorFunc = regError, ops = (1, 4)):
    # the allowed delta error and least samples after splitting the data set
    tolerateError = ops[0]; tolerateRows = ops[1]
    # stopping condition 1 : if the values of the target variable are equal, return None and generate leaf node
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1 :
        return None, leafFunc(dataSet)
    n = shape(dataSet)[1]
    error = errorFunc(dataSet)
    minError = inf; bestFeatIdx = 0; bestFeatVal = 0
    for featIndex in range(n - 1) :
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]) :
            mat0, mat1 = binarySplitDS(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolerateRows) or (shape(mat1)[0] < tolerateRows) :
                continue
            curError = errorFunc(mat0) + errorFunc(mat1)
            if curError < minError :
                bestFeatIdx = featIndex
                bestFeatVal = splitVal
                minError = curError
    # stopping condition 2 : if the delta error are too small (than the tolerated minimum delta error)
    if (error - minError) < tolerateError :
        return None, leafFunc(dataSet)
    # stopping condition 3 : if the samples in divided data set are smaller than the tolerated least number of samples 
    mat0, mat1 = binarySplitDS(dataSet, bestFeatIdx, bestFeatVal)
    if (shape(mat0)[0] < tolerateRows) or (shape(mat1)[0] < tolerateRows) :
        return None, leafFunc(dataSet)
    return bestFeatIdx, bestFeatVal

'''
Create CART
'''
def createTree(dataSet, leafType = regLeaf, errorType = regError, ops = (1, 4)):
    feature, value = chooseBestSplit(dataSet, leafType, errorType, ops)
    if feature == None :
        return value
    returnTree = {}
    returnTree['splitIndex'] = feature
    returnTree['splitValue'] = value
    leftSet, rightSet = binarySplitDS(dataSet, feature, value)
    returnTree['leftChild'] = createTree(leftSet, leafType, errorType, ops)
    returnTree['rightChild'] = createTree(rightSet, leafType, errorType, ops)
    return returnTree

'''
Judge if obj is a tree
'''
def isTree(obj):
    return type(obj).__name__ == 'dict'

'''
Transform the tree into a value (mean of the tree)
'''
def getMean(tree):
    if isTree(tree['leftChild']) :
        tree['leftChild'] = getMean(tree['leftChild'])
    if isTree(tree['rightChild']) :
        tree['rightChild'] = getMean(tree['rightChild'])
    return (tree['leftChild'] + tree['rightChild']) / 2.0

'''
Backwards pruning of the tree with test data
'''
def prune(tree, testData):
    if (shape(testData)[0] == 0) :
        return getMean(tree)
    if (isTree(tree['leftChild']) or isTree(tree['rightChild'])) :
        lSet, rSet = binarySplitDS(testData, tree['splitIndex'], tree['splitValue'])
    if (isTree(tree['leftChild'])) :
        tree['leftChild'] = prune(tree['leftChild'], lSet)
    if (isTree(tree['rightChild'])) :
        tree['rightChild'] = prune(tree['rightChild'], rSet)
    if not isTree(tree['leftChild']) and not isTree(tree['rightChild']) :
        lSet, rSet = binarySplitDS(testData, tree['splitIndex'], tree['splitValue'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['leftChild'], 2)) + sum(power(rSet[:, -1] - tree['rightChild'], 2))
        treeMean = (tree['leftChild'] + tree['rightChild']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge :
            return treeMean
        else :
            return tree
    else :
        return tree

'''
Generate linear model for leaf node in model tree
'''
def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    X[:, 1 : n] = dataSet[:, 0 : n-1]; Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0 :
        raise NameError('This matrix is singular, cannot do inverse, try increasing the second value of ops')
    weights = xTx.I * (X.T * Y)
    return weights

'''
Generate leaf node of model tree, the node is a linear model 
'''
def modelLeaf(dataSet):
    ws = linearSolve(dataSet)
    return ws

'''
Error estimation of model tree, the error value is the sum of squared error
'''
def modelError(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    X[:, 1 : n] = dataSet[:, 0 : n-1]; Y = dataSet[:, -1]
    ws = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

'''
Evaluate value of a sample (Xi) with regression tree leaf node (model, total variance value)
'''
def regTreeEval(model, Xi):
    return float(model)

'''
Evaluate value of a sample (Xi) with model tree leaf node (model, weights)
'''
def modelTreeEval(model, Xi):
    n = shape(Xi)[1]
    # when Xi contains the label in the last column
    #X = mat(ones((1, n)))
    #X[:, 1 : n] = Xi[:, 0 : n-1]
    X = mat(ones((1, n+1)))
    X[:, 1 : n+1] = Xi
    print model, Xi, X
    return float(X * model)

'''
Forecast the value of a sample with a tree regression model and correspond value evaluation method
'''
def treeForecast(tree, inData, modelEval = regTreeEval):
    if not isTree(tree) :
        return modelEval(tree, inData)
    if inData[0, tree['splitIndex']] > tree['splitValue'] :
        if isTree(tree['leftChild']) :
            return treeForecast(tree['leftChild'], inData, modelEval)
        else :
            return modelEval(tree['leftChild'], inData)
    else :
        if isTree(tree['rightChild']) :
            return treeForecast(tree['rightChild'], inData, modelEval)
        else :
            return modelEval(tree['rightChild'], inData)

'''
Forecast all test samples
'''
def createForecast(tree, testData, modelEval = regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m) :
        yHat[i, 0] = treeForecast(tree, mat(testData[i]), modelEval)
    return yHat

def testModel(): 
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    tree = createTree(trainMat, ops = (1, 20))
    yHat = createForecast(tree, testMat)
    print "R^2 value (correlation coefficient) of regression tree : ", corrcoef(yHat, testMat[:, 1], rowvar = 0)[0, 1]
    
    modelTree = createTree(trainMat, modelLeaf, modelError, (1, 20))
    yHat = createForecast(modelTree, testMat, modelTreeEval)
    print "R^2 value (correlation coefficient) of model tree : ", corrcoef(yHat, testMat[:, 1], rowvar = 0)[0, 1]
    
    weights = linearSolve(trainMat)
    for i in range(shape(testMat)[0]) :
        yHat[i] = testMat[i, 0] * weights[1, 0] + weights[0, 0]
    print "R^2 value (correlation coefficient) of linear regression : ", corrcoef(yHat, testMat[:, 1], rowvar = 0)[0, 1]
    

def main():
    dataSet = loadDataSet('ex3.txt')
    dataMat = mat(dataSet)
    #tree = createTree(dataMat, ops = (0,1))
    tree = createTree(dataMat, modelLeaf, modelError, ops = (1,10))
    print tree
    '''
    testData = loadDataSet('ex2test.txt')
    testMat = mat(testData)
    myTree = prune(tree, testMat)
    print myTree
    '''