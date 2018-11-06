import numpy as np
import sys
import os
from DataManipulations import *
from funcs import *
from MST import *


if __name__ == "__main__":

    # print(sys.argv)
    if len(sys.argv) != 4:
        raise ImportError("Not enough arguments")

    if not os.path.exists(sys.argv[1]):
        raise FileNotFoundError("train file does not exist")

    if not os.path.exists(sys.argv[2]):
        raise FileNotFoundError("test file does not exist")

    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    option = sys.argv[3]

    trainData, trainMeta = readData(trainFile)
    testData, testMeta = readData(testFile)


    if option == 'n':
        # this option implements a bayes network that uses naive bayes
        # printNB_graph(trainMeta)

        # count the probability of each Y value
        p_Y = getProbabilityDistribution(trainData, trainMeta, 'class')
        # count the probability for each feature given Y
        p_XgY = getP_XgY(trainData, trainMeta)

        # for the given test file, predict using naive bayes
        y_predicted, y_prob = predictNaiveBayes(testData, testMeta,  p_Y, p_XgY)
        # print the predictions
        print('\n')
        printPredictions(list(testData['class']), y_predicted, y_prob)

    elif option == 't':
        # this option implements a bayes network that uses TAN
        # for this first step, we need to compute the I (information gain) value
        # for each combination of the X value.
        # We will need to compute a bunch of probabilities before we can proceed with making the
        # bayesian network itself.

        # This function will return an NxN matrix which contains the information gain
        # between every pair of features
        I = getInfoGain(trainData, trainMeta)

        # build MST
        MST = prims(I, trainMeta)

        # print the MST
        printTAN(MST)

        # build conditional probability tables
        CPT = buildCPT(trainData, trainMeta, MST)

        # create predictions using constructed probability tables

    else:
        print("Invalid option")
        exit(0)

