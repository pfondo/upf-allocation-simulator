#!/usr/bin/env python3

"""
UPF allocation simulator

@ author:
    Pablo Fondo-Ferreiro <pfondo@gti.uvigo.es>
    David Candal-Ventureira <dcandal@gti.uvigo.es>
"""

import argparse
import time
import networkx as nx
from random import random, seed, gauss, sample
import statistics
from math import log10
from numpy import array as np_array, mean as np_mean, percentile as np_percentile, vstack as np_vstack
from scipy.stats import gaussian_kde
from sys import stderr, stdout

from sklearn import cluster
from sklearn.cluster import KMeans

import community as community_louvain
import itertools
from networkx.algorithms.community.centrality import girvan_newman as girvan_newman

from scipy.stats import sem as st_sem, t as st_t

DEBUG = False
PL_THRESHOLD = 2
DISTANCE_THRESHOLD = 500

DEFAULT_MIN_UPF=1
DEFAULT_MAX_UPF=10

DEFAULT_ITERATION_DURATION = 5 # In seconds

class UE:
    def __init__(self, x=0, y=0, bs=None):
        self._x = float(x)
        self._y = float(y)
        self._bs = bs
        if self._bs is not None:
            self._bs.add_UE(self)

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_coords(self):
        return [self._x, self._y]

    def set_coords(self, x, y):
        self._x, self._y = (float(x), float(y))

    def get_bs(self):
        return self._bs

    def set_bs(self, bs):
        if self._bs is not None:
            self._bs.remove_UE(self)
        if bs is not None:
            bs.add_UE(self)
        self._bs = bs

    def get_pl(self):
        if self._bs is None:
            return float("inf")
        else:
            distance = self._bs.get_distance_coords(self._x, self._y)
            return compute_path_loss(distance)

    def update_bs(self, bs, pl=float("inf")):
        if bs is None or pl + PL_THRESHOLD < self.get_pl():
            self.set_bs(bs)

    def update(self, x, y, bs, pl=float("inf")):
        self.set_coords(x, y)
        self.update_bs(bs, pl)

    def update_unconditional(self, x, y, bs):
        self.set_coords(x, y)
        self.set_bs(bs)


class BS:
    def __init__(self, id_, x, y, UPF=False):
        self._id = int(id_)
        self._x = float(x)
        self._y = float(y)
        self._UPF = UPF
        self.UEs = []

    def get_id(self):
        return self._id

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_coords(self):
        return [self._x, self._y]

    def has_UPF(self):
        return self._UPF

    def set_UPF(self, UPF):
        self._UPF = UPF

    def add_UE(self, ue):
        self.UEs.append(ue)

    def remove_UE(self, ue):
        self.UEs.remove(ue)

    def get_numUEs(self):
        return len(self.UEs)
    
    def clear_UEs(self):
        self.UEs = []

    def get_distance(self, bs2):
        return ((bs2.get_x()-self.get_x())**2 + (bs2.get_y()-self.get_y())**2)**0.5

    def get_distance_coords(self, x, y):
        return ((x-self.get_x())**2 + (y-self.get_y())**2)**0.5


''' Generates a set of connected components by conecting all the base stations
    that are positioned less than DISTANCE_THRESHOLD meters apart
'''


def generate_graph(bs_file):
    G = nx.Graph()
    BSs = {}
    highest_bs_id = -1

    with open(bs_file) as f:
        for _, line in enumerate(f):
            bs_data = line.strip().split()
            bs = BS(bs_data[0], bs_data[1], bs_data[2])
            BSs[bs.get_id()] = bs
            if bs.get_id() > highest_bs_id:
                highest_bs_id = bs.get_id()
            G.add_node(bs)
            for other_bs in G.nodes:
                if other_bs is not bs and bs.get_distance(other_bs) < DISTANCE_THRESHOLD:
                    G.add_edge(bs, other_bs)

    join_components(G)

    G_shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    return G, BSs, G_shortest_path_lengths, highest_bs_id


''' Connects the connected components to compose a single giant component.
    This procedure is performed as follow: On each iteration, from the second
    biggest connected component, the node which is at the shortest distance
    from one of the nodes of the giant component is determined, and an edge
    between these two nodes.

    This way, on each iteration the connected components are joined to the giant
    component in order of size yo achieve a single giant component.
'''


def join_components(G):
    while True:
        connected_components = list(nx.connected_components(G))
        if len(list(connected_components)) < 2:
            break
        connected_components = sorted(
            connected_components, key=len, reverse=True)

        bs1 = bs2 = distance = None
        for bs in connected_components[1]:
            for bs_giant_component in connected_components[0]:
                d = bs.get_distance(bs_giant_component)
                if distance is None or d < distance:
                    distance = d
                    bs1 = bs
                    bs2 = bs_giant_component
        G.add_edge(bs1, bs2)


''' Determines the nearest BS to the coordinates x and y
'''


def get_optimal_bs(BSs, x, y):
    distance = None
    bs = None
    for node in BSs.values():
        d = node.get_distance_coords(x, y)
        if distance is None or d < distance:
            distance = d
            bs = node
    return bs, distance


def compute_path_loss(distance):
    #p1 = 46.61
    #p2 = 3.63
    #std = 9.83
    # return p1 + (p2 * 10 * log10(distance)) + gauss(0, std)
    return 46.61 + (3.63 * 10 * log10(distance)) + gauss(0, 9.83)


''' Generates the list of UEs of the new iteration: updates data from previous
    iteration, removes UEs that do not appear in the new stage and adds those
    that did not appear in the last stage.
'''


def read_UE_data(ue_file, BSs, iteration_duration):
    UEs_last_iteration = {}

    first_timestamp_iteration = None

    UEs_new_iteration = {}

    with open(ue_file) as f:
        for line in f:
            # Read UEs from new iteration
            line = line.strip().split()
            timestamp = int(line[0])
            id_ = int(line[1].split("_")[0].split("#")[0])
            x = float(line[2])
            y = float(line[3])
            speed = float(line[4])  # Unused
            pl = None
            if len(line) > 5:
                bs = BSs[int(line[5])]
            else:
                bs, distance = get_optimal_bs(BSs, x, y)
                pl = compute_path_loss(distance)

            if first_timestamp_iteration == None:
                first_timestamp_iteration = timestamp

            if timestamp - first_timestamp_iteration > iteration_duration:
                # Iteration finished: Yield results
                for ue in [ue for id_, ue in UEs_last_iteration.items() if id_ not in UEs_new_iteration.keys()]:
                    ue.update_bs(None)

                UEs_last_iteration = UEs_new_iteration
                yield UEs_new_iteration

                # Resumed execution for next iteration: Initialize values for this iteration
                UEs_new_iteration = {}
                first_timestamp_iteration = timestamp

            # Update UE already present in previous iteration
            if id_ in UEs_last_iteration:
                ue = UEs_last_iteration[id_]
                if pl:
                    ue.update(x, y, bs, pl)
                else:
                    ue.update_unconditional(x, y, bs)
            # Only the last appearance of each UE in the iteration is considered
            elif id_ in UEs_new_iteration:
                ue = UEs_new_iteration[id_]
                if pl:
                    ue.update(x, y, bs, pl)
                else:
                    ue.update_unconditional(x, y, bs)
            # Se crea un nuevo UE
            else:
                ue = UE(x, y, bs)
            UEs_new_iteration[id_] = ue

# Deprecated: Used to generate synthetic data
def generate_UE_data_random(BSs):
    UEs_last_iteration = {}

    for i in range(100):
        UEs_new_iteration = {}
        for j in range(10000):
            id_ = j  # int(j*random())
            x = (4503.09786887-28878.1970746)*random()+28878.1970746
            y = (3852.34416744-36166.012178)*random()+36166.012178
            bs, distance = get_optimal_bs(BSs, x, y)
            pl = compute_path_loss(distance)

            if id_ in UEs_last_iteration:
                ue = UEs_last_iteration[id_]
                ue.update(x, y, bs, pl)
            elif id_ in UEs_new_iteration:
                ue = UEs_new_iteration[id_]
                ue.update(x, y, bs, pl)
            else:
                ue = UE(x, y, bs)
            UEs_new_iteration[id_] = ue
        for ue in [ue for id_, ue in UEs_last_iteration.items() if id_ not in UEs_new_iteration.keys()]:
            ue.update_bs(None)

        UEs_last_iteration = UEs_new_iteration
        yield UEs_new_iteration


def UPF_allocation_random(G, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set(
        sample([x for x in range(G.number_of_nodes())], num_UPFs))
    return BSs_with_UPF_ids

def analyze_allocation(G: nx.Graph, BSs_with_UPF_ids):
    # Used for printing nodes and analyzing the allocations
    import matplotlib.pyplot as plt


    x = []
    y = []

    colors = []

    bs : BS
    for bs in G.nodes():
        if bs.get_numUEs() == 0:
            continue
        for _ in range(bs.get_numUEs()):
            x.append(bs.get_x() / 1000)
            y.append(bs.get_y() / 1000)
            colors.append(bs.get_numUEs())

    #plt.scatter(x, y, c=colors)

    xy = np_vstack([x,y])
    z = gaussian_kde(xy)(xy)

    #sc = plt.scatter(x, y, c=z, s=10)

    hb = plt.hexbin(x, y, gridsize=10, cmap = "Blues")

    plt.xlabel("x coordinate (km)")
    plt.ylabel("y coordinate (km)")

    plt.colorbar(hb)

    x2 = []
    y2 = []
    for bs in G.nodes():
        if bs.get_id() in BSs_with_UPF_ids:
            x2.append(bs.get_x() / 1000)
            y2.append(bs.get_y() / 1000)

    plt.scatter(x2, y2, color = ['r' for x in range(len(BSs_with_UPF_ids))], marker= 'x')

    plt.show()


def UPF_allocation_greedy_percentile(G: nx.Graph, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set()

    # Inf value initialization
    latencies_list = [
        G.number_of_nodes() + 1 for _ in range(highest_bs_id + 1)]
    num_ues_list = [0 for _ in range(highest_bs_id + 1)]
    tot_ues = 0
    for bs in G.nodes:
        num_ues_list[bs.get_id()] = bs.get_numUEs()
        tot_ues += bs.get_numUEs()

    done_BSs = [False for _ in range(highest_bs_id + 1)]

    for _ in range(num_UPFs):
        best_bs = None
        best_latency = None
        best_acc_latency = None
        for bs in G.nodes:
            if not done_BSs[bs.get_id()]:
                # Check latency if bs is selected
                acc_latency = 0
                latencies_current_list = list(latencies_list)
                for bs2 in G.nodes:
                    new_latency = min(
                        G_shortest_path_lengths[bs2][bs], latencies_current_list[bs2.get_id()])
                    latencies_current_list[bs2.get_id()] = new_latency
                    acc_latency += new_latency

                # Calculate 90th percentile of ue latency
                latency = None
                acc_num_ues = 0
                for lat, num_ues_bs in sorted(zip(latencies_current_list, num_ues_list), key=lambda x: x[0], reverse=True):
                    acc_num_ues += num_ues_bs
                    if acc_num_ues >= 0.1 * tot_ues:
                        latency = lat
                        break

                assert(latency != None)

                if best_bs == None or latency < best_latency or (latency == best_latency and acc_latency < best_acc_latency):
                    best_bs = bs
                    best_latency = latency
                    best_acc_latency = acc_latency

        BSs_with_UPF_ids.add(best_bs.get_id())
        done_BSs[best_bs.get_id()] = True
        for bs2 in G.nodes:
            new_latency = G_shortest_path_lengths[bs2][best_bs]
            if new_latency < latencies_list[bs2.get_id()]:
                latencies_list[bs2.get_id()] = new_latency

    return BSs_with_UPF_ids


def UPF_allocation_greedy_percentile_fast(G: nx.Graph, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set()

    # Inf value initialization
    latencies_list = [
        G.number_of_nodes() + 1 for _ in range(highest_bs_id + 1)]
    num_ues_list = [0 for _ in range(highest_bs_id + 1)]
    tot_nodes = G.number_of_nodes()

    done_BSs = [False for _ in range(highest_bs_id + 1)]

    for _ in range(num_UPFs):
        best_bs = None
        best_latency = None
        best_acc_latency = None
        for bs in G.nodes:
            if not done_BSs[bs.get_id()]:
                # Check latency if bs is selected
                acc_latency = 0
                latencies_current_list = list(latencies_list)
                for bs2 in G.nodes:
                    new_latency = min(
                        G_shortest_path_lengths[bs2][bs], latencies_current_list[bs2.get_id()])
                    latencies_current_list[bs2.get_id()] = new_latency
                    acc_latency += new_latency

                # Calculate 90th percentile of ue latency
                latency = None
                acc_num_ues = 0
                for lat in sorted(latencies_current_list, reverse=True):
                    acc_num_ues += 1
                    if acc_num_ues >= 0.1 * tot_nodes:
                        latency = lat
                        break

                assert(latency != None)

                if best_bs == None or latency < best_latency or (latency == best_latency and acc_latency < best_acc_latency):
                    best_bs = bs
                    best_latency = latency
                    best_acc_latency = acc_latency

        BSs_with_UPF_ids.add(best_bs.get_id())
        done_BSs[best_bs.get_id()] = True
        for bs2 in G.nodes:
            new_latency = G_shortest_path_lengths[bs2][best_bs]
            if new_latency < latencies_list[bs2.get_id()]:
                latencies_list[bs2.get_id()] = new_latency

    return BSs_with_UPF_ids

# Greedy implementation iteratively picking the eNB which reduces the average latency the most
def UPF_allocation_greedy_average(G: nx.Graph, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set()

    # Inf value initialization
    latencies_list = [
        G.number_of_nodes() + 1 for _ in range(highest_bs_id + 1)]
    num_ues_list = [0 for _ in range(highest_bs_id + 1)]
    tot_ues = 0
    for bs in G.nodes:
        num_ues_list[bs.get_id()] = bs.get_numUEs()
        tot_ues += bs.get_numUEs()

    done_BSs = [False for _ in range(highest_bs_id + 1)]

    for _ in range(num_UPFs):
        best_bs = None
        best_latency = None
        for bs in G.nodes:
            if not done_BSs[bs.get_id()]:
                # Check latency if bs is selected
                acc_latency = 0
                for bs2 in G.nodes:
                    new_latency = min(
                        G_shortest_path_lengths[bs2][bs], latencies_list[bs2.get_id()])
                    acc_latency += (new_latency * num_ues_list[bs2.get_id()])

                # Calculate average latency
                latency = acc_latency / tot_ues

                if best_bs == None or latency < best_latency:
                    best_bs = bs
                    best_latency = latency

        BSs_with_UPF_ids.add(best_bs.get_id())
        done_BSs[best_bs.get_id()] = True
        for bs2 in G.nodes:
            new_latency = G_shortest_path_lengths[bs2][best_bs]
            if new_latency < latencies_list[bs2.get_id()]:
                latencies_list[bs2.get_id()] = new_latency

    return BSs_with_UPF_ids


# Greedy implementation iteratively picking the eNB which reduces the max latency the most; in case of equal max latency the one which reduces the average latency the most
def UPF_allocation_greedy_max(G: nx.Graph, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set()

    # Inf value initialization
    latencies_list = [
        G.number_of_nodes() + 1 for _ in range(highest_bs_id + 1)]
    num_ues_list = [0 for _ in range(highest_bs_id + 1)]
    tot_ues = 0
    for bs in G.nodes:
        num_ues_list[bs.get_id()] = bs.get_numUEs()
        tot_ues += bs.get_numUEs()

    done_BSs = [False for _ in range(highest_bs_id + 1)]

    for _ in range(num_UPFs):
        best_bs = None
        best_max_latency = None
        best_avg_latency = None
        for bs in G.nodes:
            if not done_BSs[bs.get_id()]:
                # Check latency if bs is selected
                acc_latency = 0
                max_latency = 0
                for bs2 in G.nodes:
                    new_latency = min(
                        G_shortest_path_lengths[bs2][bs], latencies_list[bs2.get_id()])
                    acc_latency += (new_latency * num_ues_list[bs2.get_id()])
                    if new_latency > max_latency:
                        max_latency = new_latency

                # Calculate average latency
                avg_latency = acc_latency / tot_ues

                if best_bs == None or max_latency < best_max_latency or (max_latency == best_max_latency and avg_latency < best_avg_latency):
                    best_bs = bs
                    best_max_latency = max_latency
                    best_avg_latency = avg_latency

        BSs_with_UPF_ids.add(best_bs.get_id())
        done_BSs[best_bs.get_id()] = True
        for bs2 in G.nodes:
            new_latency = G_shortest_path_lengths[bs2][best_bs]
            if new_latency < latencies_list[bs2.get_id()]:
                latencies_list[bs2.get_id()] = new_latency

    return BSs_with_UPF_ids

# Implementation of kmeans considering active eNBs, greedily selecting the best eNB in each cluster
def UPF_allocation_kmeans_greedy_average(G: nx.Graph, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set()

    # Calculate k-means clustering considering active BSs
    features = []
    BSs = []
    for bs in G.nodes:
        if bs.get_numUEs() > 0:
            features.append([bs.get_x(), bs.get_y()])
            BSs.append(bs)

    kmeans = KMeans(
        init="k-means++", # "random" / "k-means++"
        n_clusters=min(num_UPFs, len(features)),
        n_init=2,
        max_iter=1000,
        random_state=0 # To allow for reproducibility
    )

    # In case scaling wants to be enabled
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(features)
    # kmeans.fit(scaled_features)

    kmeans.fit(features)

    cluster_BS_ids_list = [[] for _ in range(num_UPFs)]
    for i in range(len(kmeans.labels_)):
        cluster_BS_ids_list[kmeans.labels_[i]].append(i)

    # Greedily pick the best UPF in each cluster
    for cluster in range(num_UPFs):
        #cluster_BS_ids = [x for x in range(len(kmeans.labels_)) if kmeans.labels_[x] == cluster]
        cluster_BS_ids = cluster_BS_ids_list[cluster]

        best_bs = None
        best_acc_latency = None
        for bs_index in cluster_BS_ids:
            bs = BSs[bs_index]
            # Check latency if bs is selected
            acc_latency = 0
            for bs2_index in cluster_BS_ids:
                bs2 = BSs[bs2_index]
                new_latency = G_shortest_path_lengths[bs2][bs]
                acc_latency += (new_latency * bs2.get_numUEs())

            # Calculate average latency
            if best_bs == None or acc_latency < best_acc_latency:
                best_bs = bs
                best_acc_latency = acc_latency

        if best_bs != None:
            BSs_with_UPF_ids.add(best_bs.get_id())

    # Add UPFs until reaching desired number of UPFs (for addressing corner cases with very few active base stations)
    if len(BSs_with_UPF_ids) < num_UPFs:
        BSs_with_UPF_ids.update(set(sample([x for x in range(G.number_of_nodes()) if x not in BSs_with_UPF_ids], num_UPFs - len(BSs_with_UPF_ids))))

    assert(len(BSs_with_UPF_ids) == num_UPFs)

    return BSs_with_UPF_ids


# Implementation of kmeans considering active BSs
def UPF_allocation_kmeans(G: nx.Graph, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set()

    ## Calculate k-means clustering considering active BSs
    features = []
    for bs in G.nodes:
        if bs.get_numUEs() > 0:
            features.append([bs.get_x(), bs.get_y()])

    kmeans = KMeans(
        init="k-means++", # "random" / "k-means++"
        n_clusters=min(num_UPFs, len(features)),
        n_init=2,
        max_iter=1000,
        random_state=0 #To allow for reproducibility
    )

    # In case scaling want to be applied
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(features)
    # kmeans.fit(scaled_features)

    kmeans.fit(features)

    # Pick UPFs closer to cluster centers
    done_BSs = [False for _ in range(highest_bs_id + 1)]

    #for center_x, center_y in scaler.inverse_transform(kmeans.cluster_centers_):
    for center_x, center_y in kmeans.cluster_centers_:
        best_UPF = None
        best_distance = None
        for bs in G.nodes:
            if not done_BSs[bs.get_id()]:
                distance = bs.get_distance_coords(center_x, center_y)
                if best_UPF == None or distance < best_distance:
                    best_UPF = bs.get_id()
                    best_distance = distance

        BSs_with_UPF_ids.add(best_UPF)
        done_BSs[best_UPF] = True

    # Add UPFs until reaching desired number of UPFs (for addressing corner cases with very few active BSs)
    if len(BSs_with_UPF_ids) < num_UPFs:
        BSs_with_UPF_ids.update(set(sample([x for x in range(G.number_of_nodes()) if x not in BSs_with_UPF_ids], num_UPFs - len(BSs_with_UPF_ids))))

    assert(len(BSs_with_UPF_ids) == num_UPFs)

    return BSs_with_UPF_ids

# Implementation of clustering based on community detection (Louvain modularity maximization) considering active BSs, greedily selecting the best eNB in each cluster
def UPF_allocation_modularity_greedy_average(G: nx.Graph, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set()

    # Louvain modularity maximization using active BSs
    G_active = nx.Graph(G)
    for bs in G.nodes:
        if bs.get_numUEs() == 0:
            G_active.remove_node(bs)

    partition = community_louvain.best_partition(G_active)

    cluster_BSs_list = [[] for _ in range(num_UPFs)]
    for bs, cluster in partition.items():
        # NOTE: Cluster joining can be improved
        cluster_BSs_list[cluster % num_UPFs].append(bs)

    # Greedily pick the best UPF in each cluster
    for cluster in range(num_UPFs):
        cluster_BS_ids = cluster_BSs_list[cluster]

        best_bs = None
        best_acc_latency = None
        for bs in cluster_BS_ids:
            # Check latency if bs is selected
            acc_latency = 0
            for bs2 in cluster_BS_ids:
                new_latency = G_shortest_path_lengths[bs2][bs]
                acc_latency += (new_latency * bs2.get_numUEs())

            # Calculate average latency
            if best_bs == None or acc_latency < best_acc_latency:
                best_bs = bs
                best_acc_latency = acc_latency

        if best_bs != None:
            BSs_with_UPF_ids.add(best_bs.get_id())

    # Add UPFs until reaching desired number of UPFs
    if len(BSs_with_UPF_ids) < num_UPFs:
        BSs_with_UPF_ids.update(set(sample([x for x in range(G.number_of_nodes()) if x not in BSs_with_UPF_ids], num_UPFs - len(BSs_with_UPF_ids))))

    assert(len(BSs_with_UPF_ids) == num_UPFs)

    return BSs_with_UPF_ids

# Implementation of clustering based on hierarchical community detection (Girvan-Newman) considering all eNBs, greedily selecting the best eNB in each cluster
def UPF_allocation_girvan_newman_greedy_average(G: nx.Graph, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set()

    ## Girvan-Newman modularity maximization considering active eNBs
    G_active = nx.Graph(G)
    for bs in G.nodes:
        if bs.get_numUEs() == 0:
            G_active.remove_node(bs)

    comp = girvan_newman(G_active)
    limited = itertools.takewhile(lambda c: len(c) <= num_UPFs, comp)
    cluster_BSs_list = [[] for _ in range(num_UPFs)]
    for communities in limited:
        for i in range(len(communities)):
            cluster_BSs_list[i] = list(communities[i])

    assert(len(cluster_BSs_list) == num_UPFs)

    #print([len(x) for x in cluster_BSs_list])


    # Greedily pick the best UPF in each cluster
    for cluster in range(num_UPFs):
        cluster_BS_ids = cluster_BSs_list[cluster]

        best_bs = None
        best_acc_latency = None
        for bs in cluster_BS_ids:
            # Check latency if bs is selected
            acc_latency = 0
            for bs2 in cluster_BS_ids:
                new_latency = G_shortest_path_lengths[bs2][bs]
                acc_latency += (new_latency * bs2.get_numUEs())

            # Calculate average latency
            if best_bs == None or acc_latency < best_acc_latency:
                best_bs = bs
                best_acc_latency = acc_latency

        if best_bs != None:
            BSs_with_UPF_ids.add(best_bs.get_id())

    # Add UPFs until reaching desired number of UPFs
    if len(BSs_with_UPF_ids) < num_UPFs:
        BSs_with_UPF_ids.update(set(sample([x for x in range(G.number_of_nodes()) if x not in BSs_with_UPF_ids], num_UPFs - len(BSs_with_UPF_ids))))

    assert(len(BSs_with_UPF_ids) == num_UPFs)

    return BSs_with_UPF_ids

''' Determines in which BS a UPF is instantiated:
    Generic function which calls the specific allocation algorithm
'''
def UPF_allocation(algorithm, G, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id, generate_new_allocation=True):
    BSs_with_UPF_ids = set()

    if generate_new_allocation or BSs_with_UPF_previous == None or len(BSs_with_UPF_previous) != num_UPFs:
        BSs_with_UPF_ids = globals()["UPF_allocation_{}".format(algorithm)](G, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id)
    else:
        for bs in BSs_with_UPF_previous:
            BSs_with_UPF_ids.add(bs.get_id())

    if DEBUG:
        print(BSs_with_UPF_ids, file=stderr)

    return BSs_with_UPF_ids

# Metrics
def get_minimum_hops_from_BS_to_UPF(G, bs, BSs_with_UPF, G_shortest_path_lengths):
    if bs.has_UPF():
        return 0, bs

    hops = None
    bs_with_upf = None
    for other_bs in BSs_with_UPF:
        try:
            # h = len(nx.shortest_path(G, source=bs,
            #                          target=other_bs)) - 1  # Dijkstra
            # Pre-computed Floyd-Wharsall
            h = G_shortest_path_lengths[bs][other_bs] - 1
        except:
            continue
        if hops is None or h < hops:
            hops = h
            bs_with_upf = other_bs

    if hops is None:
        raise Exception("No reachable UPF from BS {}".format(bs.get_id()))
    return hops, bs_with_upf


''' Returns a list with the number of hops to the nearest UPF for each UE
    NOTE: The indexes of the list do not correspond to the IDs of the UEs
'''
def get_UE_hops_list(G, BSs_with_UPF, G_shortest_path_lengths):
    UE_hops_list = []
    for bs in G.nodes:
        num_UEs = bs.get_numUEs()
        if num_UEs < 1:
            continue
        hops, _ = get_minimum_hops_from_BS_to_UPF(
            G, bs, BSs_with_UPF, G_shortest_path_lengths)
        UE_hops_list.extend([hops]*num_UEs)
    return UE_hops_list

# Used for debugging
def print_statistics(UE_hops_list, file=stdout):
    if file != None:
        print("Minimum: {}".format(min(UE_hops_list)), file=file)
        print("Maximum: {}".format(max(UE_hops_list)), file=file)
        print("Mean: {}".format(statistics.mean(UE_hops_list)), file=file)
        if len(UE_hops_list) > 1:
            print("Variance: {}".format(
                statistics.variance(UE_hops_list)), file=file)
            print("Standard deviation: {}".format(
                statistics.stdev(UE_hops_list)), file=file)


def mean_confidence_interval(data, confidence=0.95):
    if (min(data) == max(data)):
        m = min(data)
        h = 0
    else:
        a = 1.0*np_array(data)
        n = len(a)
        m, se = np_mean(a), st_sem(a)
        h = se * st_t._ppf((1+confidence)/2., n-1)
    return (m, max(m-h, 0), m+h)


def main():
    seed(0)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", help="Specifies the UPF allocation algorithm [Supported: random/greedy_percentile/greedy_percentile_fast/greedy_average/greedy_max/kmeans/kmeans_greedy_average/modularity_greedy_average/girvan_newman_greedy_average].", required=True)
    parser.add_argument(
        "--minUPFs", help="Specifies the minimum number of UPFs to be allocated [Default: {}].".format(DEFAULT_MIN_UPF), type=int, default=DEFAULT_MIN_UPF)
    parser.add_argument(
        "--maxUPFs", help="Specifies the maximum number of UPFs to be allocated [Default: {}].".format(DEFAULT_MAX_UPF), type=int, default=DEFAULT_MAX_UPF)
    parser.add_argument(
        "--bsFile", help="File containing the information about the base stations [Format: each line contains the id, x coordinate and y coordinate of a base station separated by spaces].", required=True)
    parser.add_argument(
        "--ueFile", help="File containing the information about the users throughout the simulation [Format: each line contains the timestamp, user id, x coordinate, y coordinate, speed and, optionally, the base station id to which the user is attached].", required=True)
    parser.add_argument(
        "--iterationDuration", help="Duration of each time-slot [Default: {}].".format(DEFAULT_ITERATION_DURATION), type=int, default=DEFAULT_ITERATION_DURATION)
    args = parser.parse_args()
    algorithm = args.algorithm
    min_UPFs = args.minUPFs
    max_UPFs = args.maxUPFs
    ue_file = args.ueFile
    bs_file = args.bsFile
    iteration_duration = args.iterationDuration

    # Generate graph
    G, BSs, G_shortest_path_lengths, highest_bs_id = generate_graph(bs_file)

    for num_UPFs in range(min_UPFs, max_UPFs + 1):
        list_results_num_hops = []
        list_results_elapsed_time = []

        global cluster_BSs_list
        if "kmeans" in algorithm:
            # Calculate k-means clustering considering all BSs
            features = []
            for bs in G.nodes:
                features.append([bs.get_x(), bs.get_y()])
            global kmeans 
            kmeans = KMeans(
                init="k-means++", # "random" / "k-means++"
                n_clusters=min(num_UPFs, len(features)),
                n_init=2,
                max_iter=1000,
                random_state=0 #To allow for reproducibility
            )

            kmeans.fit(features)

        elif "girvan_newman" in algorithm:
            comp = girvan_newman(G)
            limited = itertools.takewhile(lambda c: len(c) <= num_UPFs, comp)

            cluster_BSs_list = [[] for _ in range(num_UPFs)]
            for communities in limited:
                for i in range(len(communities)):
                    cluster_BSs_list[i] = communities[i]
        elif "modularity" in algorithm:
            partition = community_louvain.best_partition(G)

            cluster_BSs_list = [[] for _ in range(num_UPFs)]
            for bs, cluster in partition.items():
                # NOTE: Cluster joining can be improved
                cluster_BSs_list[cluster % num_UPFs].append(bs)

        for _ in range(1):
            # Clear UEs in BSs
            for bs in G.nodes():
                bs.clear_UEs()

            iteration = 0
            # First UPF allocation is random
            BSs_with_UPF_ids = UPF_allocation(
                "random", G, num_UPFs, None, G_shortest_path_lengths, highest_bs_id)

            BSs_with_UPF = []

            for bs in G.nodes:
                if bs.get_id() in BSs_with_UPF_ids:
                    bs.set_UPF(True)
                    BSs_with_UPF.append(bs)
                else:
                    bs.set_UPF(False)


            upf_start_time = time.process_time()  # seconds
            print("Running {} for {} UPFs".format(
                algorithm, num_UPFs), file=stderr)
            for UEs in read_UE_data(ue_file, BSs, iteration_duration):
                BSs_with_UPF_previous = BSs_with_UPF
                iteration += 1
                UE_hops_list = get_UE_hops_list(
                    G, BSs_with_UPF, G_shortest_path_lengths)

                start_time = time.process_time() * 1e3  # milliseconds
                BSs_with_UPF_ids = UPF_allocation(
                    algorithm, G, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id)
                end_time = time.process_time() * 1e3  # milliseconds
                elapsed_time = end_time - start_time

                BSs_with_UPF = []

                for bs in G.nodes:
                    if bs.get_id() in BSs_with_UPF_ids:
                        bs.set_UPF(True)
                        BSs_with_UPF.append(bs)
                    else:
                        bs.set_UPF(False)

                if DEBUG:
                    print("UE hops to UPF: {}".format(
                        UE_hops_list), file=stderr)
                    print_statistics(UE_hops_list, file=stderr)
                    print("\n\n", file=stderr)

                num_hops_90th = np_percentile(UE_hops_list, 90)
                list_results_num_hops.append(num_hops_90th)
                list_results_elapsed_time.append(elapsed_time)

                print("\r  Iteration {}: {} -> {} UEs".format(
                    iteration, len(UEs), int(num_hops_90th)), end='', file=stderr)

        print("\r  Number of iterations: {}".format(iteration), file=stderr)

        upf_end_time = time.process_time()  # seconds
        upf_elapsed_time = upf_end_time - upf_start_time

        print("  Elapsed time: {:.3f} seconds".format(
            upf_elapsed_time), file=stderr)

        mci_num_hops = mean_confidence_interval(list_results_num_hops, 0.95)
        mci_elapsed_time = mean_confidence_interval(
            list_results_elapsed_time, 0.95)

        print("{} {} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(algorithm,
                                                                       num_UPFs, mci_num_hops[0], mci_num_hops[
                                                                           1], mci_num_hops[2], mci_elapsed_time[0],
                                                                       mci_elapsed_time[1], mci_elapsed_time[2]))


if __name__ == "__main__":
    main()
