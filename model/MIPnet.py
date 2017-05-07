import networkx as nx
import numpy as np
from collections import Counter, defaultdict

"""
This file implements the MIPnet model which contains the graph of users and repos
"""

class MIPnet():
    def __init__(self, P, R, decay = 0.9):
        self.P = P # partner
        self.R = R # repos
        self.mip = nx.MultiGraph()
        self.mip.add_nodes_from(self.P, ntype="user")
        self.mip.add_nodes_from(self.R, ntype="repo")
        self.centrality = {} # degree
        self.userconnectedness = {} # weighted
        self.userconnectedness2 = {} # number
        self.prevContent = Counter()
        self.prevDesign = Counter()
        self.prevConsume = Counter()
        self.userEncoding = {p:np.zeros(5) for p in self.P}

        self.decay = decay


    # currently, user-user, repo-repo increase by max 1 in weight,
    # could also be 1 per common edit
    def update_edges_for_time(self ,ints):
        thisBucketUserRepos = defaultdict(set)
        thisBucketRepoUsers = defaultdict(set)
        # create or update user-repo edges and repo-repo edges
        for ix ,i in enumerate(ints):
            cuser, crepo, ctype, cowner = i
            # Update Counts for User
            if ctype == 'content':
                self.prevContent[cuser] += 1
            elif ctype == 'design':
                self.prevDesign[cuser] += 1
            elif ctype == 'consume':
                self.prevDesign[cuser] += 1

            # repo-repo: create/update edge if not exists in current time
            if crepo not in thisBucketUserRepos[cuser] and len(thisBucketUserRepos[cuser]) > 0:
                for connectRepo in thisBucketUserRepos[cuser]:
                    if self.mip.has_edge(crepo, connectRepo):
                        self.mip[crepo][connectRepo][0]['weight' ] +=1
                    else:
                        self.mip.add_edge(crepo, connectRepo, weight=1, ntype='r-r')
            thisBucketUserRepos[cuser].add(crepo)

            # user-user, similar to repo-repo
            if cuser not in thisBucketRepoUsers[crepo] and len(thisBucketRepoUsers[crepo]) > 0:
                for connectUser in thisBucketRepoUsers[crepo]:
                    if self.mip.has_edge(cuser, connectUser):
                        self.mip[cuser][connectUser][0]['weight' ] +=1
                    else:
                        self.mip.add_edge(cuser, connectUser, weight=1, ntype='u-u')
            thisBucketRepoUsers[crepo].add(cuser)

            # user-repo
            edge_exists = False
            # check whether edge exists
            if (cuser, crepo, None) in self.mip.edges(cuser ,crepo):
                # if yes, iterate over edges to find whether the correct edge exists
                for ednum ,val in self.mip[cuser][crepo].iteritems():
                    if val['ntype'] == ctype:
                        edge_exists = True
                        self.mip[cuser][crepo][ednum]['weight' ] +=1
            if not edge_exists:
                self.mip.add_edge(cuser, crepo, weight=1, ntype=ctype, owner=cowner)
        self.centrality = nx.degree_centrality(self.mip)
        self.updateUserConnectedness()

    def updateUserConnectedness(self):
        """
        Computes the average weight between users
        """
        for r in self.R:
            current_weight = 0.
            number_connected = 0.
            # Step 1: Get all users connected to a repo
            U = [user for user, edge in self.mip[r].iteritems() if
                 edge[0]['ntype'] in {'design', 'content', 'consume'}]
            # Step 2: Get all edges between the users
            for u1 in range(len(U ) -1):
                for u2 in range( u1 +1, len(U)):
                    try:
                        current_weight += self.mip[U[u1]][U[u2]][0]['weight']
                        number_connected += 1
                    except:
                        pass # If users have not edited in same time delta
            # Step 3: Get number of possible connections
            possible_connections = len(U) * (len(U) + 1) / 2
            # Step 4: Compute C1 and C2
            if possible_connections == 0:
                self.userconnectedness[r] = 0
                self.userconnectedness2[r] = 0
            else:
                self.userconnectedness[r] = current_weight / possible_connections * 100
                self.userconnectedness2[r] = number_connected / possible_connections * 100


    def decay_weights(self):
        for u ,v ,d in self.mip.edges(data=True):
            d['weight'] = d['weight'] * self.decay
