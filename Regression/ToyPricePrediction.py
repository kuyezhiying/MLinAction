"""
Predict the price of LEGO toys
"""

from time import sleep
from numpy import mat, ones, shape
import json
import urllib2
import regression

def searchGoods(retX, retY, numSet, year, numPieces, originPrice):
    sleep(10)
    myAPIStr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?\
                key=%s&country=US&q=lego+%d&alt=json' % (myAPIStr, numSet)
    pg = urllib2.urlopen(searchURL)
    retDict = json.load(pg.read())
    for i in range(len(retDict['items'])) :
        try :
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new' :
                newFlag = 1
            else :
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv :
                sellingPrice = item['price']
                if sellingPrice > originPrice * 0.5 :
                    print "%d\t%d\t%d\t%f\t%f" % (year, numPieces, newFlag, originPrice, sellingPrice)
                retX.appen([year, numPieces, newFlag, originPrice])
                retY.appen(sellingPrice)
        except :
            print 'problem with item %d' % i

def setDataCollect(retX, retY):
    searchGoods(retX, retY, 8288, 2006, 800, 49.99)
    searchGoods(retX, retY, 10030, 2002, 3096, 269.99)
    searchGoods(retX, retY, 10179, 2007, 5195, 499.99)
    searchGoods(retX, retY, 10181, 2007, 3428, 199.99)
    searchGoods(retX, retY, 10189, 2008, 5922, 299.99)
    searchGoods(retX, retY, 10196, 2009, 3263, 249.99)

def testLego():
    legoX = []; legoY = []
    setDataCollect(legoX, legoY)
    m, n = shape(legoX)
    xMat = mat(ones((m, n + 1)))
    xMat[:, 1:5] = mat(legoX)
    weights = regression.standardRegression(legoX, legoY)
    print weights

if __name__ == '__main__':
    testLego()