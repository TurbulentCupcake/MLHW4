import numpy as np
import sys
import os
from DataManipulations import *
from funcs import *

if __name__ == "__main__":

    print(sys.argv)
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
        printNB_graph(trainMeta)

        # count the probability of each Y value
        p_Y = getProbabilityDistribution(trainData, trainMeta, 'class')
        # count the probability for each feature given Y
        p_XgY = getP_XgY(trainData, trainMeta)
        # for the given test file, predict using naive bayes
        y_predicted, y_prob = predictNaiveBayes(testData, testMeta,  p_Y, p_XgY)







        pass
    elif option == 't':
        # this option implements a bayes network that uses TAN
        pass
    else:
        print("Invalid option")
        exit(0)

