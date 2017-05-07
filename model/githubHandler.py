import math
import numpy as np
import networkx as nx
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

"""
Defines a bunch of helper functions to handle data

"""

def get_all_interactions_for_timebucket(t, data, label_encodings):


    #creates a list of time
    counter = 0
    interactions = [] #user, repo, type
    for ix,key in data.iteritems():
        try: # there is one empty set in the data...
            for time, types in zip(key['time_buckets'], key['actions']):
                if time == t:
                    if not label_encodings[types] == "none":
                        interactions.append((ix[0],
                                             ix[1],
                                             label_encodings[types],
                                             key['user_is_owner']))
                #print types, time,
            counter +=1
        except:
            pass
    print "found", len(interactions), "interactions"
    return interactions

def constructAutoEncoderData(udata, label_encs, current_set = 8):
    action2idx = {"content_same": 0,
                  "design_same": 1,
                  "consume_same": 2,
                  "content_diff": 3,
                  "design_diff": 4,
                  "consume_diff": 5,
                  }

    all_lengths = []
    for ukey, u in udata.iteritems():
        current_acts = sum([1 if t <= current_set else 0 for t in u['time_buckets']])
        all_lengths.append(current_acts)
    # Make it so that most sequences are included, rest will be cut off
    input_len = int(np.mean(all_lengths) * 1.5)
    print input_len, "Number of sequence length for autoencoder"
    # 1. Construct padded input

    user_enc_name = []
    x = []
    for ukey, u in udata.iteritems():
        # Here will be change once we have diff/same data
        x.append(np.array(([action2idx[label_encs[act] + "_same"] for act in u['actions']
                            if label_encs[act] != "none"][:input_len])))
        user_enc_name.append(ukey)
    x = sequence.pad_sequences(x, maxlen=input_len)

    x = np.array(x)
    y = to_categorical(x, num_classes=None).reshape(len(x), input_len, -1)
    return user_enc_name, x,y


def connectionWeight(mip, n1,n2, gammas):
    '''
    Compute the connections weight for all interaction types
    '''
    weights = np.zeros(4)
    for ednum,val in mip[n1][n2].iteritems():
        weights[0] += val['weight'] * gammas[0][val['ntype']]
        weights[1] += val['weight'] * gammas[1][val['ntype']]
        weights[2] += val['weight'] * gammas[2][val['ntype']]
        weights[3] += val['weight'] * gammas[3][val['ntype']]
    return weights


def adamicAdarProximity(mip, s, t, gammas):
    """
    Compute the adamicAdarProximity given two nodes
    """

    proximity = [0.,0., 0., 0.]
    if (mip.has_node(s)==False) | (mip.has_node(t)==False):
        return [0.,0., 0., 0.]
    for node in nx.common_neighbors(mip, s, t):
        weights = connectionWeight(mip, s,node, gammas) + connectionWeight(mip, t,node, gammas)
        if weights[0]!=0: # 0 essentially means no connection
            # gives more weight to "rare" shared neighbors, adding small number to avoid dividing by zero
            proximity = proximity + (weights*(1/(math.log(mip.degree(node, weight = 'weight'))+1e-9)))
    return proximity

def computeUserInCommonWeight(mip, u,r):
    """
    given two nodes, computes the weight that users have in common
    """
    w = 0.0
    for u2 in mip[r].keys():
        try:
            w += mip[u][u2][0]['weight']
        except:
            pass
    return w

def computeDOI(mip, u, r, params, alpha, beta):
    # precomputed centrality + scaling factor
    API = mip.centrality[r] #* 100
    prox = adamicAdarProximity(mip.mip,u,r, params)
    #print API, prox
    #print prox
    return alpha * API + beta * prox[0], API, prox

def get_X_features(mip, mipnet, user, repo, mask, params):
    x = []
    if mask[0]: # API from MIP-DOI
        x += [mip.centrality[repo] * 100] # scaled to %
    D = adamicAdarProximity(mipnet, user, repo, params)
    if mask[1]: # Distance all
        x += [D[0]]
    if mask[2]: # Distance content
        x += [D[1]]
    if mask[3]: # Distance design
        x += [D[2]]
    if mask[4]: # Distance Consume
        x += [D[3]]
    if mask[5]: # Weighted User Connectedness
        x += [mip.userconnectedness[repo]]
    if mask[6]: # Fraction User Connectedness
        x += [mip.userconnectedness2[repo]]
    if mask[7]: # Stars / Forks
        pass # TODO
    if mask[8]: # User Embeddings
        x += list(mip.userEncoding[user])
    if mask[8]: # Owner yes/no
        try:
            if mipnet[user][repo][0]['owner']:
                x += [1.,0.]
            else:
                x += [0.,1.]
        except:
            x += [0.,1.]
    if mask[9]: # previous interaction count content
        x += [mip.prevContent[user]]
    if mask[10]: # previous interaction count design
        x += [mip.prevDesign[user]]
    if mask[11]: # previous interaction count consume
        x += [mip.prevConsume[user]]
    if mask[12]: # Total weight between user and users in common
        x += [computeUserInCommonWeight(mipnet, user,repo)]
    return np.array(x, dtype=np.float32)


def construct_X_large(Y, mip, mipnet, users, repos, mask, params, user2idx, repo2idx, samplesize=300):
    '''
    Constructs a feature vector for each user,
    for the positives as well as large_samplesize other repos
    '''
    reposet = set(repos)
    X_large = dict()
    for ix, u in enumerate(users):
        targets = Y[u].keys()
        # Create first samples from uniform dist over all repos
        samplerepos = np.random.choice(list(reposet - set(targets)),
                                       samplesize, replace=False)

        true_vects = []
        false_vects = []
        for r in targets:
            # print u,r
            true_vects.append((get_X_features(mip, mipnet, u, r, mask, params), repo2idx[r]))
        for r in samplerepos:
            false_vects.append((get_X_features(mip, mipnet, u, r, mask, params), repo2idx[r]))

        X_large[user2idx[u]] = (true_vects, false_vects)
        if ix % 5 == 0:
            print ix
    return X_large


