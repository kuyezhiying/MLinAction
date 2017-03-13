from numpy import array, tile, zeros
from os import listdir
import operator, PreProcess

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    
    # compute distance
    diffMatrix = tile(inX, (dataSetSize, 1)) - dataSet
    squareDiffMat = diffMatrix ** 2
    squareDistance = squareDiffMat.sum(axis = 1)
    distances = squareDistance ** 0.5
    
    sortedDistIndicies = distances.argsort()
    classCount = {}
    
    # select the top-k points with minor distance
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    
    # sort labels with class count
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    
    return sortedClassCount[0][0]


def datingClassTest():
    # the ratio of test set over the whole data set
    holdOutRatio = 0.20
    
    datingDataMat, datingLabels = PreProcess.file2matrix('datingTestSet.txt')
    normMat = PreProcess.autoNorm(datingDataMat)
    
    rows = normMat.shape[0]
    numTestVecs = int(rows * holdOutRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs : rows, :], datingLabels[numTestVecs : rows], 4)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    
    # user inputs
    videoPercent = float(raw_input("percentage of time spent playing video games?"))
    flyMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCreamLiter = float(raw_input("liters of ice cream consumed per year?"))
    
    inArr = array([videoPercent, flyMiles, iceCreamLiter])
    
    datingDatMat, datingLabels = PreProcess.file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = PreProcess.autoNorm(datingDatMat)
    
    classifierResult = classify((inArr - minVals) / ranges, normMat, datingLabels, 3)
       
    print "You will probably like this person: ", resultList[classifierResult - 1]

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    rows = len(trainingFileList)
    trainingMat = zeros((rows, 1024))
    # store handwriting digits in training set to a matrix
    for i in range(rows):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        digitClass = int(fileStr.split('_')[0])
        hwLabels.append(digitClass)
        trainingMat[i, :] = PreProcess.img2vector('trainingDigits/%s' %fileNameStr)
    
    errorCount = 0.0
    testFileList = listdir('testDigits')
    testRows = len(testFileList)
    # classify the handwriting digits in test set and compute error rate
    for i in range(testRows):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        testDigitClass = int(fileStr.split('_')[0])
        testDigitVector = PreProcess.img2vector('testDigits/%s' %fileNameStr)
        # classify test case
        classifierResult = classify(testDigitVector, trainingMat, hwLabels, 3)        
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, testDigitClass)
        if (classifierResult != testDigitClass):
            errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(testRows))


if __name__ == '__main__':
    '''
    group, labels = createDataSet()
    result = classify([0, 0], group, labels, 3)
    print result
    datingClassTest()
    '''
    #classifyPerson()
    handwritingClassTest()