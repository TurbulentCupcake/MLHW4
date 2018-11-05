

import numpy as np
from collections import Counter
from DataManipulations import *
import pandas as pd
import math

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
        freq_table = getFrequencyTable(data, meta, feature)

        # create a probability distribution table for each feature
        Y = [y.encode(encoding='UTF-8') for y in list(meta._attributes['class'][1])]
        X = [x.encode(encoding='UTF-8') for x in list(meta._attributes[feature][1])]

        # create a table that holds the frequency
        pd_table = pd.DataFrame(np.zeros(len(Y) * len(X)).reshape((len(X), len(Y))),
                             columns=Y, index=X)

        for y in Y:
            for x in X:
                pd_table[y].loc[x] = (freq_table[y].loc[x])/(freq_table[y].sum()) # likelihood with laplace smoothing

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

    Y = [y.encode(encoding='UTF-8') for y in list(meta._attributes['class'][1])]
    predictions = [] # holds the predictions
    p_probabilities = [] # holds the confidence of the predictions
    for i,xdata in enumerate(data):

        # build a dictionary to store the probabilities for each class
        class_probs = dict()
        for y in Y:
            class_probs[y] = 0

        for y in Y:
            p_temp = p_Y[y]
            for x in meta.names()[0:(len(meta.names())-1)]:
                p_temp *= p_XgY[x][y].loc[xdata[x]]

            class_probs[y] = p_temp

        # add up the each of the class_probs
        total_prob_sums = 0
        for c in class_probs.values(): total_prob_sums+=c

        final_class_probs = list(class_probs.values())
        for j in range(len(final_class_probs)): final_class_probs[j] /= total_prob_sums

        max_prob_loc = np.argmax(final_class_probs)
        predictions.append(Y[max_prob_loc])
        p_probabilities.append(final_class_probs[max_prob_loc])

    return predictions, p_probabilities


def printPredictions(actual, predictions, probabilities):

    match_count = 0
    for a, p, prob in zip(actual, predictions, probabilities):
        print(str(a,'utf-8'), str(p,'utf-8'), "{0:.12f}".format(prob))
        if a == p: match_count+=1

    print('\n')
    print(match_count)



def getFrequencyTable(data, meta, feature):

    Y = [y.encode(encoding='UTF-8') for y in list(meta._attributes['class'][1])]
    X = [x.encode(encoding='UTF-8') for x in list(meta._attributes[feature][1])]

    # create a table that holds the frequency
    table = pd.DataFrame(np.zeros(len(Y)*len(X)).reshape((len(X), len(Y))),
                         columns=Y, index=X)

    # fill in the count for each occurance
    for y, x in zip(data['class'],data[feature]):
            table[y].loc[x]+=1

    # add one laplace smoothing
    for y in Y:
        for x in X:
            table[y].loc[x]+=1

    return table



def getFrequencyTable_crossFeature(data, meta, feature1, feature2):
    """
    Overloaded function to count frequency given two features
    Laplacian smoothing already accounted for within this function.
    :param data: dataset
    :param meta: metadata
    :param feature1: first feature
    :param feature2: second feature
    :return: matrix of size NxM where N is the number of possible values in first feature and
    M is the number of possible values in the second feature.

    """
    Y = [y.encode(encoding='UTF-8') for y in list(meta._attributes['class'][1])]
    X1 = [x.encode(encoding='UTF-8') for x in list(meta._attributes[feature1][1])]
    X2 = [x.encode(encoding='UTF-8') for x in list(meta._attributes[feature2][1])]

    table = pd.DataFrame(np.zeros(len(X1)*len(X2)).reshape((len(X1), len(X2))),
                         columns=X2, index=X1)

    for xi, xj in zip(data[feature1], data[feature2]):
        table[xj].loc[xi]+=1

    # add 1 for laplacian smoothing
    for xi in X1:
        for xj in X2:
            table[xj].loc[xi]+=1

    return table


def getProbabilityDistribution_crossFeature(freq_table):
    """
    Returns a probability distribution table for a given frequency distribution
    :param freq_table: table of frequency distribution
    :return:  probability distribution matrix of same dimensions as input
    """

    pd_table = pd.DataFrame(np.zeros(freq_table.shape),
                            columns=freq_table.columns, index = freq_table.index)

    for i in freq_table.columns:
        for j in freq_table.index:
            pd_table[i].loc[j] = freq_table[i].loc[j]/freq_table.values.sum()

    return pd_table


def getIndexSplit(data, meta, feature):

    X1 = [x.encode(encoding='UTF-8') for x in list(meta._attributes[feature][1])]

    splits = dict()
    for val in X1:
        splits[val] = []

    for i,d in enumerate(data):
        splits[d[feature]].append(i)

    return splits




def getInfoGain(data, meta):
    """
    This function is used to compute the p_XXgY for a given
    dataset
    :param data: dataset
    :param meta: metadata
    :return: NxN matrix where each position gives us the information gain between
    any two features
    """

    # First, we need to split the dataset by the labels.
    # Once we have done that, we will need to compute the combination
    # off occurances of every value from each split dataset
    # I.e we need to build a frequency table where we count the
    # number of times a combination of values from a pair of
    # features occurs. Then we need to perform additive smoothing (Laplacian).
    # What would this look like? For every subset of the dataset split on value
    # We would need take a pair of features and build a NxM matrix depending
    # on the number of features in each of those selected pair of features
    # Then as we see a pair of features, we would need to add +1 to the
    # position in that matrix that corresponds to that combination of values
    # in tha matrix. Finally, we need to add 1 to make laplacian smoothing
    # Then we need to divide each entry of the matrix with the sum of all
    # counts in the matrix. This will give you the desired XXgY for a given

    # obtain all the possible Y values and values for each feature in the dataset
    Y = [y.encode(encoding='UTF-8') for y in list(meta._attributes['class'][1])]
    X = dict()
    for feature in meta.names()[0:(len(meta.names())-1)]:
        X[feature] = [x.encode(encoding='UTF-8') for x in list(meta._attributes[feature][1])]


    # create a matrix to hold the information gain between each set
    infoGain = pd.DataFrame(np.zeros(len(X.keys())*len(X.keys())).reshape((len(X.keys()), len(X.keys()))),
                            columns=list(X.keys()), index=list(X.keys()))

    P_XgY = getP_XgY(data, meta)
    P_Y = getProbabilityDistribution(data, meta, 'class')

    # outer loop iterates through every x_i value
    for x_i in X.keys():

        # inner loop iterates through every x_j value
        for x_j in X.keys():

            if x_i != x_j:


                # iterate through every possible y value in order to
                # split the dataset appropriately

                P_XXgY = dict()

                y_splits = getIndexSplit(data, meta, 'class') # obtain all splits of the data
                for y in Y:

                #########################  This code gives us P(xi,xj | y) ###################

                    # create a frequency table that accounts for the number of
                    # occurances of two pairs of features
                    freq_table = getFrequencyTable_crossFeature(data[y_splits[y]], meta, x_i, x_j)

                    # compute the probability distribution for the frequency table above
                    pd_table = getProbabilityDistribution_crossFeature(freq_table)

                    P_XXgY[y] = pd_table

                ################################################################################

                # obtain P(Xi|Y) and P(Xj|Y) for both features
                P_XigY = P_XgY[x_i]
                P_XjgY = P_XgY[x_j]

                # now, we must obtain p(xi,xj,y) for every y.
                # We know that P(xi,xj|y) = P(xi,xj,y)/P(y)
                # Thus, we can figure out P(xi,xj,y) = P(xi,xj,y)P(y)
                # We can use our pre-calculated values for conditional probabilities
                # to get the conditional probabilities for each given y.

                ######################## This computes P(xi,xj,y) ##############################
                P_XXY = dict()

                for y in Y:

                    table = pd.DataFrame(np.zeros(P_XXgY[y].shape), columns=P_XXgY[y].columns, index = P_XXgY[y].index)

                    for v1 in X[x_i]:
                        for v2 in X[x_j]:

                            table[v2].loc[v1] = P_XXgY[y][v2].loc[v1]*P_Y[y]

                    P_XXY[y] = table

                ################################################################################

                # Now we have obtained P_XXgY, P_XXY, PXigY and PXjgY
                # It is time to compute? I

                I_value = 0
                for v1 in X[x_i]:
                    for v2 in X[x_j]:
                        for y in Y:
                            I_value += P_XXY[y][v2].loc[v1]*\
                                       math.log((P_XXgY[y][v2].loc[v1]/(P_XigY[y].loc[v1]*P_XjgY[y].loc[v2])), 2)

                infoGain[x_i].loc[x_j] = I_value

    return infoGain












































    