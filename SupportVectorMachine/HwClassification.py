'''
Handwriting classification
'''

from os import listdir
from numpy import zeros, mat, shape, nonzero, multiply, sign
import svm

'''
Transform the digit(32*32 binary image) to vector(1*1024) 
'''
def img2vector(filename):
    returnVec = zeros((1, 1024))
    imgReader = open(filename)
    for i in range(32):
        line = imgReader.readline()
        for j in range(32):
            returnVec[0, 32 * i + j] = int(line[j])
    return returnVec 

'''
Load images from a directory
'''
def loadImages(dirName):
    hwLables = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m) :
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9 :
            hwLables.append(-1)
        else :
            hwLables.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLables

'''
Classify digits with rbf kernel
'''
def testDigits(kTup = ('rbf', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    bias, alphas = svm.plattSMO(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    svIdx = nonzero(alphas.A > 0)[0]
    sv = dataMat[svIdx]; svLabels = labelMat[svIdx]
    print "there are %d support vectors." % shape(sv)[0]
    m = shape(dataMat)[0]
    errorCount = 0
    for i in range(m) :
        kernelEval = svm.kernelTrans(sv, dataMat[i, :], kTup)
        predictLabel = kernelEval.T * multiply(svLabels, alphas[svIdx]) + bias
        if sign(predictLabel) != sign(labelArr[i]) :
            errorCount += 1
    print "the training error rate is : %f" % (float(errorCount) / m)
    dataArr, labelArr = loadImages('testDigits')
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    m = shape(dataMat)[0]
    errorCount = 0
    for i in range(m) :
        kernelEval = svm.kernelTrans(sv, dataMat[i, :], kTup)
        predictLabel = kernelEval.T * multiply(svLabels, alphas[svIdx]) + bias
        if sign(predictLabel) != sign(labelArr[i]) :
            errorCount += 1
    print "the test error rate is : %f" % (float(errorCount) / m)


if __name__ == '__main__':
    testDigits()