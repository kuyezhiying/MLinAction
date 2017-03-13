from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def createDataSetFromFile(filename):
    fr = open(filename)
    lenses = [line.strip().split('\t') for line in fr.readlines()]
    features = ['age', 'prescript', 'astigmatic', 'tearRate']
    return lenses, features

'''
Compute the entropy of the whole data set : 
H = -sigma(p(xi)*log2(p(xi))), i = 1 ... n, n is the number of labels.
'''
def computeEntropy(dataSet):
    numEntries = len(dataSet)
    labelCountMap = {}
    for featureVec in dataSet:
        curLabel = featureVec[-1]
        if curLabel not in labelCountMap.keys():
            labelCountMap[curLabel] = 0
        labelCountMap[curLabel] += 1
    entropy = 0.0
    for key in labelCountMap:
        prob = float(labelCountMap[key]) / numEntries
        entropy -= prob * log(prob, 2)
    return entropy

'''
Divide the data set by a specific feature and a feature value
'''
def splitDataSet(dataSet, featureNo, value):
    retDataSet = []
    for item in dataSet:
        if item[featureNo] == value:
            extractedFeatureVec = item[:featureNo]
            extractedFeatureVec.extend(item[featureNo + 1 :])
            retDataSet.append(extractedFeatureVec)
    return retDataSet

'''
Choose the best feature to split the data set
'''
def chooseBestFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = computeEntropy(dataSet)
    bestInfoGain = 0.0; bestFeatureNo = -1
    # for each feature, compute the information gain
    for i in range(numFeatures):
        # select the column (all feature values) below feature i
        featureList = [item[i] for item in dataSet]
        # select all unique feature values
        uniqueFeatureVals = set(featureList)
        featureEntropy = 0.0
        # for each unique feature value, split the data set and compute entropy
        for value in uniqueFeatureVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            featureEntropy += prob * computeEntropy(subDataSet)
        # compute information gain with the feature i
        infoGain = baseEntropy - featureEntropy
        if (infoGain > bestInfoGain) :
            bestInfoGain = infoGain
            bestFeatureNo = i
    return bestFeatureNo

'''
Choose the majority class in a class list
'''
def majorityCnt(classList):
    classCountDict = {}
    for vote in classList:
        if vote not in classCountDict.keys():
            classCountDict[vote] = 0
        classCountDict[vote] += 1
    sortedClassCount = sorted(classCountDict.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

'''
Create decision tree
'''
def createTree(dataSet, labels):
    classList = [item[-1] for item in dataSet]
    # stopping criteria 1 : labels in the class list are all the same
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # stopping criteria 2 : All the features are traversed, then the majority of the label in the class list are returned
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # choose the best feature
    bestFeature = chooseBestFeature(dataSet)
    bestFeatureLabel = labels[bestFeature]
    # construct decision tree
    decisionTree = {bestFeatureLabel: {}}
    del(labels[bestFeature])
    # get all values of the best fetaure
    featureValues = [item[bestFeature] for item in dataSet]
    uniqueFeatVals = set(featureValues)
    for val in uniqueFeatVals:
        subLabels = labels[:]
        decisionTree[bestFeatureLabel][val] = createTree(splitDataSet(dataSet, bestFeature, val), subLabels)
    return decisionTree 

'''
Use pickle module to store the constructed decision tree (serialize the tree object (dictionary) to file)
'''
def storeTree(tree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(tree, fw)
    fw.close()
    
'''
Deserialize the tree object from file
'''
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def classify(tree, featureLabels, testVector):
    root = tree.keys()[0]
    subTree = tree[root]
    # find the index of current node (feature) label in feature set
    featureIndex = featureLabels.index(root)
    for key in subTree.keys():
        if testVector[featureIndex] == key :
            if type(subTree[key]).__name__ == 'dict' :
                classLabel = classify(subTree[key], featureLabels, testVector)
            else :
                classLabel = subTree[key]
    return classLabel


if __name__ == '__main__':
    #dataSet, labels = createDataSet()
    dataSet, labels = createDataSetFromFile('lenses.txt')
    '''
    entropy = computeEntropy(dataSet)
    print entropy
    dataSet[0][-1] = 'maybe'
    print dataSet
    entropy = computeEntropy(dataSet)
    print entropy
    retDataSet = splitDataSet(dataSet, 0, 0)
    print retDataSet
    retDataSet = splitDataSet(dataSet, 0, 1)
    print retDataSet
    '''
    '''
    bestFeatureNo = chooseBestFeature(dataSet)
    print bestFeatureNo
    '''
    '''
    classList = [example[-1] for example in dataSet]
    majorityClass = majorityCnt(classList)
    print majorityClass
    '''
    myTree = createTree(dataSet, labels)
    print myTree
    
    storeTree(myTree, 'DecisionTreeObject_lense.txt')
    tree = grabTree('DecisionTreeObject_lense.txt')

    '''
    labels = ['no surfacing', 'flippers']
    result = classify(tree, labels, [1, 1])
    print result
    '''
    features = ['age', 'prescript', 'astigmatic', 'tearRate']
    result = classify(tree, features, ['young', 'hyper', 'no', 'normal'])
    print result
