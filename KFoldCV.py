from funcs import *
import random
from MST import *

def stratifyData(X, y, num_folds):
    """
    Stratifies data and returns a set of folds that can be iterated over
    :param X:
    :param y:
    :param num_folds:
    :return:
    """

    # get unique class
    classes = list(set(y))
    indices = dict()
    for c in classes:
        indices[c] = []

    # split the indices between the two classes
    for i in range(0, len(y)):
        indices[y[i]].append(i)

    # shuffle both the sets
    for c in classes:
        random.shuffle(indices[c])

    # compute data per fold
    dpf = round(len(X)/num_folds)
    # ratio of classes
    count = dict(collections.Counter(y))
    ratio = count[classes[0]]/len(y)
    # folds will hold the indices for each fold of data
    folds = dict()
    for i in range(num_folds):
        folds[i] = []

    # for each fold
    for i in range(num_folds):

        num_class_0 = round(ratio*dpf)  # number of instances to pull from class 0
        num_class_1 = dpf - num_class_0  # number of instances to pull from class 1

        if num_class_0 <= len(indices[classes[0]]) and num_class_1 <= len(indices[classes[1]]): # ensure both classes have enough data to split
            for j in range(num_class_0):
                folds[i].append(indices[classes[0]].pop())
            for j in range(num_class_1):
                folds[i].append(indices[classes[1]].pop())
        else: # if one of the classes do not have enough data, then simply place all the remaining values into a single fold
            for j in range(len(indices[classes[0]])):
                folds[i].append(indices[classes[0]].pop())
            for j in range(len(indices[classes[1]])):
                folds[i].append(indices[classes[1]].pop())

    return folds


def getAccuracy(y_predicted, y_actual):

    hit_count = 0
    for y_p, y_a in zip(y_predicted, y_actual):
        if y_p == y_a: hit_count+=1

    return hit_count/len(y_actual)


if __name__ == '__main__':

    filename = 'chess-KingRookVKingPawn.arff'
    data, meta = readData(filename)
    folds = stratifyData(data, data['class'], 10)
    folds = list(folds.values())

    NB_accuracy = np.zeros(len(folds))
    TAN_accuracy = np.zeros(len(folds))


    for i in range(len(folds)):

        print("Fold-",i)
        testData_idx = folds[i]
        trainData_idx = folds[0:i] + folds[i+1:len(folds)]
        # flatten train list
        trainData_idx = [y for x in trainData_idx for y in x]

        testData = data[testData_idx]
        trainData = data[trainData_idx]

        # first, do naive bayes and compute accuracy
        print('Computing Naive Bayes')
        # count the probability of each Y value
        p_Y = getProbabilityDistribution(trainData, meta, 'class')
        # count the probability for each feature given Y
        p_XgY = getP_XgY(trainData, meta)

        # for the given test file, predict using naive bayes
        y_predicted, y_prob = predictNaiveBayes(testData, meta,  p_Y, p_XgY)

        NB_accuracy[i] = getAccuracy(y_predicted, list(testData['class']))
        print(NB_accuracy[i])

        print('Computing TAN')
        # next, do the TAN algorithm
        I = getInfoGain(trainData, meta)

        # build MST
        MST = prims(I, meta)


        # build conditional probability tables
        CPT = buildCPT(trainData, meta, MST)

        # create predictions using constructed probability tables
        predictions, probabilities = predictTAN(testData, meta, CPT, MST)

        TAN_accuracy[i] = getAccuracy(predictions, list(testData['class']))
        print(TAN_accuracy[i])

    print(NB_accuracy)
    print(TAN_accuracy)















