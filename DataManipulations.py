# this method is used to read, write and modify data for the arff files

import scipy.io.arff as arff


def readData(filename):
    if filename == "None":
        raise Exception("Filename doesn't exist")

    data, meta = arff.loadarff(filename)

    return data, meta


def getUniqueFeatures(data, metadata):
    # create a feature map for feature -> unique attribute mapping
    features = dict()

    # initialize feature map
    for name in metadata.names():
        features[name] = []

    for i, name in enumerate(metadata.names()):
        for j in range(len(data)):
            if data[j][i] not in features[name]:
                features[name].append(data[j][i])

    return features


def getUniqueVals(data, feature):

    vals = []

    for val in data[feature]:
        if val not in vals:
            vals.append(val)

    return vals

