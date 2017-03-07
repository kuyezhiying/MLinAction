from numpy import ones, log, array

'''
A sample data set from the posting board of dalmatian (spotty dog) fans
'''
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], 
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmatian', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'], 
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], 
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 0 stands for formal speech, 1 stands for abusive speech
    classVector = [0, 1, 0, 1, 0, 1]
    return postingList, classVector

'''
Traverse the data set to generate a vocabulary of all the appearing words
'''
def createVocabList(dataSet):
    vocabSet = set([])
    for doc in dataSet :
        # the union of two sets
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)

'''
Create vector for a given document with set-of-words model (using the occurrence of a word as a feature)
'''
def doc2OneHotVec(vocabList, document):
    returnVec = [0] * len(vocabList)
    for word in document :
        if word in vocabList :
            returnVec[vocabList.index(word)] = 1
        else :
            print "the word [%s] is not in the vocabulary!" % word
    return returnVec

'''
Create vector for a given document with bag-of-words model (using the frequency of occurrence of a word as a feature)
'''
def doc2Vec(vocabList, document):
    returnVec = [0] * len(vocabList)
    for word in document :
        if word in vocabList :
            returnVec[vocabList.index(word)] += 1
    return returnVec

'''
Train NB classifier, return the condition probability p(wj/ci) and class probability p(ci)
The conditional independence assumption between features : p(w|ci) = p(w0, w1, ..., wn|ci) = p(w0|ci)p(w1|ci)...p(wn|ci)
p(wj|ci) = (#wj in all docs with class i) / (#words in all docs with class i)
p(ci) = (#docs with class i) / (number of all docs) 
'''
def trainNB(docsVecMatrix, docsCategory):
    numDocs = len(docsVecMatrix)
    pAbusive = sum(docsCategory) / float(numDocs)   #p(c1), p(c0) = 1 - p(c1)    
    numFeatures = len(docsVecMatrix[0])
    
    #initialize the word count for each class
    #p0WordCount = zeros(numFeatures); p1WordCount = zeros(numFeatures)
    p0WordCount = ones(numFeatures); p1WordCount = ones(numFeatures)    # to avoid a zero probability  
    #p0WordsNum = 0.0; p1WordsNum = 0.0
    p0WordsNum = 2.0; p1WordsNum = 2.0
    
    # count the number of appearing words and the number of total words for each class
    for i in range(numDocs) :
        if docsCategory[i] == 1 :
            p1WordCount += docsVecMatrix[i]
            p1WordsNum += sum(docsVecMatrix[i])
        else :
            p0WordCount += docsVecMatrix[i]
            p0WordsNum += sum(docsVecMatrix[i])
    # condition probability p(wj|c1)
    #p1Vector = p1WordCount / p1WordsNum
    p1Vector = log(p1WordCount / p1WordsNum)
    # condition probability p(wj|c0)
    #p0Vector = p0WordCount / p0WordsNum
    p0Vector = log(p0WordCount / p0WordsNum)
    
    return p0Vector, p1Vector, pAbusive

'''
Classify the category of test doc
'''
def classify(testDocVec, p0Vec, p1Vec, pClass1):
    p1 = sum(testDocVec * p1Vec) + log(pClass1)
    p0 = sum(testDocVec * p0Vec) + log(1 - pClass1)
    if p1 > p0 :
        return 1
    else :
        return 0

def testingNB():
    listOfPosts, listOfClasses = loadDataSet()
    vocabList = createVocabList(listOfPosts)
    trainMatrix = []
    for doc in listOfPosts :
        trainMatrix.append(doc2Vec(vocabList, doc))
    p0Vec, p1Vec, pAbusive = trainNB(trainMatrix, listOfClasses)
    
    testEntry = ['love', 'my', 'dalmatian']
    testDocVec = array(doc2Vec(vocabList, testEntry))
    print testEntry, ' classified as ', classify(testDocVec, p0Vec, p1Vec, pAbusive)
    
    testEntry = ['stupid', 'garbage']
    testDocVec = array(doc2Vec(vocabList, testEntry))
    print testEntry, ' classified as ', classify(testDocVec, p0Vec, p1Vec, pAbusive)


if __name__ == '__main__':
    '''
    listOfPosts, listOfClasses = loadDataSet()
    vocabList = createVocabList(listOfPosts)
    print vocabList
    docVector = doc2Vec(vocabList, listOfPosts[0])
    print docVector
    trainMatrix = []
    for doc in listOfPosts :
        trainMatrix.append(doc2Vec(vocabList, doc))
    p0Vec, p1Vec, pAbusive = trainNB(trainMatrix, listOfClasses)
    print p0Vec, p1Vec, pAbusive
    '''
    testingNB()