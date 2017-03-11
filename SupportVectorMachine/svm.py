from numpy import mat, shape, zeros, multiply, nonzero, exp, sign
import random

class OptStruct:
    def __init__(self, dataMat, labelMat, C, toler, kernelTuple):
        self.X = dataMat
        self.labelMat = labelMat
        self.C = C
        self.toler = toler
        self.m = shape(dataMat)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.bias = 0
        self.errorCache = mat(zeros((self.m, 2)))   # [(alpha_i, error_i)], i = 1, 2, ..., m
        self.KernelMat = mat(zeros((self.m, self.m)))
        for i in range(self.m) :
            self.KernelMat[:, i] = kernelTrans(self.X, self.X[i, :], kernelTuple)

'''
Load data and class labels from file
'''
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines() :
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

'''
Calculate the error of predicating the class of Xk
'''
def calcErrorK(opt, k):
    # before using kernel
    #fXk = float(multiply(opt.alphas, opt.labelMat).T * (opt.X * opt.X[k, :].T)) + opt.bias
    
    # using kernel
    fXk = float(multiply(opt.alphas, opt.labelMat).T * opt.KernelMat[:, k] + opt.bias)
    errorK = fXk - float(opt.labelMat[k])
    return errorK

'''
Select another alpha j which are different from alpha i from m alpha parameters in a random way
'''
def selectRandAlpha(i, m):
    j = i
    while (j == i) :
        j = int(random.uniform(0, m))
    return j

'''
Select another alpha j which has the maximum step length with alpha i 
'''
def selectAlphaJ(i, opt, errorI):
    maxK = -1; maxDeltaError = 0; errorJ = 0
    # set the bit of Xi in the error cache to be valid (= 1)
    opt.errorCache[i] = [1, errorI]
    # find all the valid (non zero) alpha parameters in the error cache
    validErrorCacheList = nonzero(opt.errorCache[:, 0].A)[0]
    if len(validErrorCacheList) > 1 :
        # find the alpha which has the maximum step length (relative error) with alpha i
        for k in validErrorCacheList :
            if k == i :
                continue
            errorK = calcErrorK(opt, k)
            deltaError = abs(errorI - errorK)
            if deltaError > maxDeltaError :
                maxK = k; maxDeltaError = deltaError; errorJ = errorK
        return maxK, errorJ
    else :  # the first time to select alpha j, only one valid alpha in the error cache
        j = selectRandAlpha(i, opt.m)
        errorJ = calcErrorK(opt, j)
    return j, errorJ

'''
Update the error of Xk in the error cacke
'''
def updateErrorK(opt, k):
    errorK = calcErrorK(opt, k)
    opt.errorCache[k] = [1, errorK]

'''
Adjust alpha value if it is higher than H or lower than L
'''
def adjustAlpha(alphaJ, H, L):
    if alphaJ > H :
        alphaJ = H
    if alphaJ < L :
        alphaJ = L
    return alphaJ

'''
Kernel transform function
'''
def kernelTrans(X, Xi, kTup):
    m = shape(X)[0]
    K = mat(zeros((m, 1)))
    if kTup[0] == 'linear' :    # linear kernel
        K = X * Xi.T
    elif kTup[0] == 'rbf' :     # radial basis kernel  K(x, y) = exp(-||x - y||^2 / (2 * delta^2))
        for j in range(m) :
            deltaRow = X[j, :] - Xi
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else :
        raise NameError('The kernel cannot be recognized.')
    return K

'''
Train SVM with a simplified SMO(Sequential Minimal Optimization) method
'''
def simpleSMO(dataArr, labelArr, C, toler, maxIter):
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    m = shape(dataMat)[0]
    alphas = mat(zeros((m, 1)))
    bias = 0; iteration = 0
    while (iteration < maxIter) :
        alphaPairsChanged = 0
        for i in range(m) :
            fXi = float(multiply(alphas, labelMat).T * (dataMat * dataMat[i, :].T)) + bias
            errorI = fXi - float(labelMat[i])
            if ((labelMat[i] * errorI < -toler) and (alphas[i] < C) or \
                ((labelMat[i] * errorI > toler) and (alphas[i] > 0))) :
                # select another alpha parameter randomly
                j = selectRandAlpha(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMat * dataMat[j, :].T)) + bias
                errorJ = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # set L and H to be in [0, C]
                if (labelMat[i] != labelMat[j]) :
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else :
                    L = max(0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])
                if L == H :
                    print "L == H"; continue
                # eta is the most optimal modifier delta of alpha j
                eta = 2.0 * dataMat[i, :] * dataMat[j, :].T - dataMat[i, :] * dataMat[i, :].T - dataMat[j, :] * dataMat[j, :].T
                if eta >= 0 :
                    print "eta >= 0"; continue
                # update the alpha pair, the delta value of difference is same, but in different direction
                alphas[j] -= labelMat[j] * (errorI - errorJ) / eta
                # adjust the value of alpha j if it is too large or too small
                alphas[j] = adjustAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001) :
                    print "alpha j not moving enough"; continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = bias - errorI - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i, :] * dataMat[i, :].T - \
                                     labelMat[j] * (alphas[j] - alphaJold) * dataMat[i, :] * dataMat[j, :].T
                b2 = bias - errorJ - labelMat[i] * (alphas[i] - alphaIold) * dataMat[i, :] * dataMat[j, :].T - \
                                     labelMat[j] * (alphas[j] - alphaJold) * dataMat[j, :] * dataMat[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]) :
                    bias = b1
                elif (0 < alphas[j]) and (C > alphas[j]) :
                    bias = b2
                else :
                    bias = (b1 + b2) / 1.0
                alphaPairsChanged += 1
                print "Iteration: %d  i: %d, pairs changed: %d" % (iteration, i, alphaPairsChanged)
        if (alphaPairsChanged == 0) :
            iteration += 1
        else :
            iteration = 0
        print "Iteration number : %d" % iteration
    return bias, alphas

'''
Inner loop part : optimize the alpha pair, returns 1 when succeed and 0 when failed
'''
def optimizeAlphaPair(i, opt):
    errorI = calcErrorK(opt, i)
    if ((opt.labelMat[i] * errorI < -opt.toler) and (opt.alphas[i] < opt.C) or \
        ((opt.labelMat[i] * errorI > opt.toler) and (opt.alphas[i] > 0))) :
        # select another alpha parameter j which has the maximum step length with alpha i
        j, errorJ = selectAlphaJ(i, opt, errorI)
        alphaIold = opt.alphas[i].copy()
        alphaJold = opt.alphas[j].copy()
        # set L and H to be in [0, C]
        if (opt.labelMat[i] != opt.labelMat[j]) :
            L = max(0, opt.alphas[j] - opt.alphas[i])
            H = min(opt.C, opt.C + opt.alphas[j] - opt.alphas[i])
        else :
            L = max(0, opt.alphas[i] + opt.alphas[j] - opt.C)
            H = min(opt.C, opt.alphas[i] + opt.alphas[j])
        if L == H :
            print "L == H"; return 0
        
        # eta is the most optimal modifier delta of alpha j
        #eta = 2.0 * opt.X[i, :] * opt.X[j, :].T - opt.X[i, :] * opt.X[i, :].T - opt.X[j, :] * opt.X[j, :].T       

        # using kernel function
        eta = 2.0 * opt.KernelMat[i, j] - opt.KernelMat[i, i] - opt.KernelMat[j, j]
        if eta >= 0 :
            print "eta >= 0"; return 0
        # update the alpha pair, the delta value of difference is same, but in different direction
        opt.alphas[j] -= opt.labelMat[j] * (errorI - errorJ) / eta
        # adjust the value of alpha j if it is too large or too small
        opt.alphas[j] = adjustAlpha(opt.alphas[j], H, L)
        updateErrorK(opt, j)
        if (abs(opt.alphas[j] - alphaJold) < 0.00001) :
            print "alpha j not moving enough"; return 0
        opt.alphas[i] += opt.labelMat[j] * opt.labelMat[i] * (alphaJold - opt.alphas[j])
        updateErrorK(opt, i)
        
        #b1 = opt.bias - errorI - opt.labelMat[i] * (opt.alphas[i] - alphaIold) * opt.X[i, :] * opt.X[i, :].T - \
        #                         opt.labelMat[j] * (opt.alphas[j] - alphaJold) * opt.X[i, :] * opt.X[j, :].T
        #b2 = opt.bias - errorJ - opt.labelMat[i] * (opt.alphas[i] - alphaIold) * opt.X[i, :] * opt.X[j, :].T - \
        #                         opt.labelMat[j] * (opt.alphas[j] - alphaJold) * opt.X[j, :] * opt.X[j, :].T
        
        # using kernel function to compute bias
        b1 = opt.bias - errorI - opt.labelMat[i] * (opt.alphas[i] - alphaIold) * opt.KernelMat[i, i] - \
                                 opt.labelMat[j] * (opt.alphas[j] - alphaJold) * opt.KernelMat[i, j]
        b2 = opt.bias - errorJ - opt.labelMat[i] * (opt.alphas[i] - alphaIold) * opt.KernelMat[i, j] - \
                                 opt.labelMat[j] * (opt.alphas[j] - alphaJold) * opt.KernelMat[j, j]
        if (0 < opt.alphas[i]) and (opt.C > opt.alphas[i]) :
            opt.bias = b1
        elif (0 < opt.alphas[j]) and (opt.C > opt.alphas[j]) :
            opt.bias = b2
        else :
            opt.bias = (b1 + b2) / 1.0
        return 1
    else :
        return 0

'''
Train SVM with Platt SMO(Sequential Minimal Optimization) method
'''
def plattSMO(dataArr, labelArr, C, toler, maxIter, kTup = ('linear', 0)):
    opt = OptStruct(mat(dataArr), mat(labelArr).transpose(), C, toler, kTup)
    iteration = 0
    entireDataSet = True; alphaPairsChanged = 0
    while (iteration < maxIter) and ((alphaPairsChanged > 0) or entireDataSet) :
        alphaPairsChanged = 0
        if entireDataSet :
            for i in range(opt.m) :
                alphaPairsChanged += optimizeAlphaPair(i, opt)
                print "Full-data-set, iter : %d  i : %d, pairs changed %d" % (iteration, i, alphaPairsChanged)
        else :
            nonBoundAlphas = nonzero((opt.alphas.A > 0) * (opt.alphas.A < C))[0]
            for i in nonBoundAlphas :
                alphaPairsChanged += optimizeAlphaPair(i, opt)
                print "Non-bound, iter : %d  i : %d, pairs changed %d" % (iteration, i , alphaPairsChanged)
        iteration += 1
        if entireDataSet :
            entireDataSet = False
        elif (alphaPairsChanged == 0) :
            entireDataSet = True
        print "Iteration number : %d" % iteration
    return opt.bias, opt.alphas

'''
Calculate weights according to the trained alpha vector
'''
def calcWeights(alphas, dataArr, labelArr):
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    weights = zeros((n, 1))
    for i in range(m) :
        weights += multiply(alphas[i] * labelMat[i], dataMat[i, :].T)
    return weights 

'''
Test RBF (radial basis function) kernel, tor is the parameter in RBF
'''
def testRBF(tor = 1.3):
    dataArr, labelArr = loadDataSet('RBFTrainingSet.txt')
    bias, alphas = plattSMO(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', tor))
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    # support vector index
    svIdx = nonzero(alphas.A > 0)[0]
    sv = dataMat[svIdx]
    svLabels = labelMat[svIdx]
    print "there are %d support vectors." % shape(sv)[0]
    m = shape(dataMat)[0]
    errorCount = 0
    for i in range(m) :
        kernelEval = kernelTrans(sv, dataMat[i, :], ('rbf', tor))
        predictLabel = kernelEval.T * multiply(svLabels, alphas[svIdx]) + bias
        if sign(predictLabel) != sign(labelArr[i]) :
            errorCount += 1
    print "the training error rate is : %f" % (float(errorCount) / m)
    dataArr, labelArr = loadDataSet('RBFTestSet.txt')
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    m = shape(dataMat)[0]
    errorCount = 0
    for i in range(m) :
        kernelEval = kernelTrans(sv, dataMat[i, :], ('rbf', tor))
        predictLabel = kernelEval.T * multiply(svLabels, alphas[svIdx]) + bias
        if sign(predictLabel) != sign(labelArr[i]) :
            errorCount += 1
    print "the test error rate is : %f" % (float(errorCount) / m)
    

if __name__ == '__main__':
    '''
    dataArr, labelArr = loadDataSet('testSet.txt')
    #bias, alphas = simpleSMO(dataArr, labelArr, 0.6, 0.001, 40)
    bias, alphas = plattSMO(dataArr, labelArr, 0.6, 0.001, 40)
    print bias, alphas[alphas > 0]
    
    weights = calcWeights(alphas, dataArr, labelArr)
    print weights
    
    dataMat = mat(dataArr)
    print dataMat[0] * mat(weights) + bias, labelArr[0]
    '''
    testRBF()