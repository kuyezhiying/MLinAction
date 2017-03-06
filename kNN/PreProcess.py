# process data

from numpy import zeros, tile, array
#import matplotlib
import matplotlib.pyplot as plt

'''
Extract features from text file to store in a matrix 
'''
def file2matrix(filename):
    fileReader = open(filename)
    lines = fileReader.readlines()
    numberOfLines = len(lines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in lines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0 : 3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1;
    return returnMat, classLabelVector

'''
Visualize the data with Matplotlib
'''
def showData(dataMat, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.scatter(dataMat[:, 1], dataMat[:, 2])
    ax.scatter(dataMat[:, 1], dataMat[:, 2], 15.0 * array(labels), 15.0 * array(labels))
    plt.show()

'''
Normailize feature values
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    
    #normDataSet = zeros(shape(dataSet))
    rows = dataSet.shape[0]
    
    # normailize
    normDataSet = dataSet - tile(minVals, (rows, 1))
    normDataSet = normDataSet / tile(ranges, (rows, 1))
    
    return normDataSet, ranges, minVals

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


if __name__ == '__main__':
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    showData(datingDataMat, datingLabels)
    
    '''
    normMat, ranges, minVals = autoNorm(datingDataMat)
    print datingDataMat, datingLabels[0 : 20]
    print normMat, ranges, minVals
    '''
    
    '''
    testVector = img2vector('testDigits/0_13.txt')
    print testVector[0, 0:31]
    '''