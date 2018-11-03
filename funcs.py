

import numpy as np
from collections import Counter
from DataManipulations import *
import pandas as pd

def printNB_graph(meta):


    names = meta.names()
    types = meta.types()

    for i in range(len(names)-1):
        if types[i] == 'nominal':
            print(names[i], names[-1])


def getProbabilityDistribution(data, meta, feature):
    """
    Returns the probability of each y value occuring
    :param data: dataset
    :param meta:
    :return: a dictionary countaining the probability distribution for value in a
                given feature
    """

    # extract values of the given feature
    if feature not in meta.names():
        raise Exception("feature name not found.")

    vals = dict(Counter(data[feature])) # compute counts of each value

    total_counts_with_psuedo = 0
    for key in vals.keys(): # add laplace smoothing count
        vals[key]+=1
        total_counts_with_psuedo += vals[key]

    # convert to probability distribution
    for key in vals.keys():
        vals[key] /= total_counts_with_psuedo


    return vals


def getP_XgY(data, meta):
    """
    obtains the probability of X given Y
    :param trainData: train dataset
    :param trainMeta: train dataset metadata
    :return: This will return a dictionary that gives us likelihood tables
    for each feature [ P(X=x|Y=y) ]
    """

    feature_probability_table = dict()

    for feature in meta.names():
        # create a frequency table for each feature
        freq_table = getFrequencyTable(data, feature)

        # create a probability distribution table for each feature
        Y = getUniqueVals(data, 'class')
        X = getUniqueVals(data, feature)

        # create a table that holds the frequency
        pd_table = pd.DataFrame(np.zeros(len(Y) * len(X)).reshape((len(X), len(Y))),
                             columns=Y, index=X)

        for y in Y:
            for x in X:
                pd_table[y].loc[x] = (freq_table[y].loc[x]+1)/(freq_table[y].sum()+len(Y)+1) # likelihood with laplace smoothing

        feature_probability_table[feature] = pd_table

    return feature_probability_table


def predictNaiveBayes(data, meta, p_Y, p_XgY):

    # get the prior probability for each possibility of Y.
    # This means that you need to go through each of the given x
    # values in the test data sample, find its prior probability for
    # each value in that feature. Then multiply all of that, then
    # multiply that with the probability of y and store that.
    # repeat this for every y value and store that sum.
    # then go through each of the probabilities and normalize them
    # then output the class of the largest one

    Y = getUniqueVals(data,'class')

    predictions = [] # holds the predictions
    p_probabilities = [] # holds the confidence of the predictions
    for i,xdata in enumerate(data):

        # build a dictionary to store the probabilities for each class
        class_probs = dict()
        for y in Y:
            class_probs[y] = 0

        for y in Y:
            p_temp = 1
            for x in meta.names():
                p_temp *= p_XgY[x][y].loc[xdata[x]]

            p_temp *= p_Y[y]

            class_probs[y] = p_temp

        # add up the each of the class_probs
        total_prob_sums = 0
        for c in class_probs: total_prob_sums+=c

        final_class_probs = class_probs.values()
        for i in range(len(final_class_probs)): final_class_probs[i] /= total_prob_sums

        max_prob_loc = np.argmax(final_class_probs)
        predictions.append(class_probs.keys()[max_prob_loc])
        p_probabilities.append(class_probs.keys()[max_prob_loc])

    return predictions, p_probabilities
    


















    pass











def getFrequencyTable(data, feature):

    Y = getUniqueVals(data, 'class')
    X = getUniqueVals(data, feature)

    # create a table that holds the frequency
    table = pd.DataFrame(np.zeros(len(Y)*len(X)).reshape((len(X), len(Y))),
                         columns=Y, index=X)


    # fill in the count for each occurance
    for y, x in zip(data['class'],data[feature]):
            table[y].loc[x]+=1

    return table
