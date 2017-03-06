import kNN

if __name__ == '__main__':
    group, labels = kNN.createDataSet()
    result = kNN.classify([0, 0], group, labels, 3)
    print result