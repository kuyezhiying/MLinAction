import matplotlib.pyplot as plt
import trees

'''
Set the style of text area and arrow
'''
decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
leafNode = dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")

'''
Plot tree node with text annotations
'''
def plotNode(nodeText, centerPoint, parentPoint, nodeType):
    createPlot.ax1.annotate(nodeText, xy = parentPoint, xycoords = 'axes fraction', \
                            xytext = centerPoint, textcoords = 'axes fraction', \
                            va = "center", ha = "center", bbox = nodeType, arrowprops = arrow_args)

'''
Plot the text annotation between parent-child nodes
'''
def plotMidText(centerPoint, parentPoint, text):
    xMid = (parentPoint[0] - centerPoint[0]) / 2.0 + centerPoint[0]
    yMid = (parentPoint[1] - centerPoint[1]) / 2.0 + centerPoint[1]
    createPlot.ax1.text(xMid, yMid, text)
    
def plotTree(tree, parentPoint, nodeText):
    numLeaves = getNumLeafNodes(tree)
    root = tree.keys()[0]
    centerPoint = (plotTree.xOff + (1.0 + float(numLeaves)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(centerPoint, parentPoint, nodeText)
    plotNode(root, centerPoint, parentPoint, decisionNode)
    subTree = tree[root]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in subTree.keys():
        if  type(subTree[key]).__name__ == 'dict':
            plotTree(subTree[key], centerPoint, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(subTree[key], (plotTree.xOff, plotTree.yOff), centerPoint, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), centerPoint, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
    
def createTreePlot(tree):
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    plotTree.totalW = float(getNumLeafNodes(tree))
    plotTree.totalD = float(getTreeDepth(tree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(tree, (0.5, 1.0), '')
    plt.show()
    
def createPlot():
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon = False)
    plotNode('Decision Node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('Leaf Node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()
    
'''
Get number of leaf nodes
'''
def getNumLeafNodes(tree):
    numLeaves = 0
    root = tree.keys()[0]
    subTree = tree[root]
    for key in subTree.keys():
        if type(subTree[key]).__name__ == 'dict':
            numLeaves += getNumLeafNodes(subTree[key])
        else:
            numLeaves += 1
    return numLeaves

'''
Get depth of the tree
'''
def getTreeDepth(tree):
    maxDepth = 0
    root = tree.keys()[0]
    subTree = tree[root]
    for key in subTree.keys():
        if type(subTree[key]).__name__ == 'dict':
            depth = getTreeDepth(subTree[key]) + 1
        else:
            depth = 1
        if depth > maxDepth :
            maxDepth = depth
    return maxDepth

def retrieveTree(i):
    foreast = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}, \
               {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
               ]
    return foreast[i]


if __name__ == '__main__':
    dataSet, labels = trees.createDataSet()
    myTree = trees.createTree(dataSet, labels)
    getNumLeafNodes(myTree)
    getTreeDepth(myTree)
    createPlot()