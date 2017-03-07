import re
import random
import bayes

def parseText(text):
    tokens = re.split(r'\W*', text)
    return [tok.lower() for tok in tokens if len(tok) > 2]

def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26) :
        wordList = parseText(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        
        wordList = parseText(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    
    # create vocabulary list of documents
    vocabList = bayes.createVocabList(docList)
    
    # split the documents into training set and test set
    trainingSet = range(50); testSet = []
    for i in range(10) :
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    
    trainMatrix = []; trainClasses = []
    for docIndex in trainingSet :
        trainMatrix.append(bayes.doc2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    
    # train NB model
    p0Vec, p1Vec, pSpam = bayes.trainNB(trainMatrix, trainClasses)
    
    # predicate and compute the error rate
    errorCount = 0
    for docIndex in testSet :
        testDocVec = bayes.doc2Vec(vocabList, docList[docIndex])
        if bayes.classify(testDocVec, p0Vec, p1Vec, pSpam) != classList[docIndex] :
            errorCount += 1
    print 'the error rate is : ', float(errorCount) / len(testSet)
    

if __name__ == '__main__':
    spamTest()