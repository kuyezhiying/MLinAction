import feedparser
import operator
import random
import bayes

def parseText(text):
    import re
    tokens = re.split(r'\W*', text)
    return [tok.lower() for tok in tokens if len(tok) > 2]

'''
Calculate the frequence of words in the vocabulary list and return the top-30 frequent words
'''
def calcMostFreqWords(vocabList, fullText, topk):
    freqDict = {}
    for token in vocabList :
        freqDict[token] = fullText.count(token)
    sortedFreqDict = sorted(freqDict.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedFreqDict[:topk]

def analyseLocalWords(feed0, feed1):
    docList = []; classList = []; fullText = []
    minLen = min(len(feed0['entries']), len(feed1['entries']))
    for i in range(minLen) :
        wordList = parseText(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    
        wordList = parseText(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        
    # create vocabulary list of documents
    vocabList = bayes.createVocabList(docList)
    
    # remove the top 30 frequent words
    top30Words = calcMostFreqWords(vocabList, fullText, 30)
    for pairW in top30Words :
        if pairW[0] in vocabList :
            vocabList.remove(pairW[0])   
    
    # split the documents into training set and test set
    trainingSet = range(2 * minLen); testSet = []
    for i in range(20) :
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
    return vocabList, p0Vec, p1Vec

def getTopWords(ny, sf):
    vocabList, pNYVec, pSFVec = analyseLocalWords(ny, sf)
    topNYWords = []; topSFWords = []
    for i in range(len(pNYVec)) :
        if pNYVec[i] > -5.0 :
            topNYWords.append((vocabList[i], pNYVec[i]))
        if pSFVec[i] > -5.0 :
            topSFWords.append((vocabList[i], pSFVec[i]))
    sortedNY = sorted(topNYWords, key = lambda pair: pair[1], reverse = True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY :
        print item[0]
    sortedSF = sorted(topSFWords, key = lambda pair: pair[1], reverse = True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF :
        print item[0]


def main():
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    #vocabList, pNY, pSF = analyseLocalWords(ny, sf)
    getTopWords(ny, sf)