

def maxKey(V, K, mstSet):

    # need to deal with this for ties
    max = -float('inf')
    max_val = None
    for v in V:
        if K[v] > max and v not in mstSet:

            max = K[v]
            max_val = v
    return max_val


def prims(I, meta):
    """
    Uses prims algorithm to compute a minimum spanning tree
    :param I: Adjacency matrix containing mutual information (?)
    :return: not sure yet lol]
    """

    # initialize all vertices
    V = meta.names()[0:(len(meta.names())-1)]
    V_index = range(len(V))

    # create a dictionary to store the next node of a given node
    P = dict()
    for v in V:
        P[v] = None

    # create a dictionary to store the weight of edge from each node
    K = dict()
    for v in V:
        K[v] = -float('inf')

    K[V[0]] = 0 # make this 0, so it is picked first
    mstSet = []


    for v in V:

        # pick the vertex with maximum weight
        u = maxKey(V, K, mstSet)

        # append the maximum weight vertex to the list
        mstSet.append(u)

        for v2 in V:
            if I[v2].loc[u] > 0 and v2 not in mstSet and K[v2] < I[v2].loc[u]:
                K[v2] = I[v2].loc[u]
                P[v2] = u


    for p in P.keys():
        print(p, P[p])









