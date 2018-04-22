import numpy as np
import torch
import pdb
import math
from collections import defaultdict, deque
from struct_gen import *

# Constants
sum_type = "sum"
prd_type = "prd"

class CVMetaData(object):
    def __init__(self, cv):
        self.depth = 0
        self.masks_by_level = []
        self.connections_by_level = []
        self.type_by_level = []

        # internal properties
        self.level_label_by_node = {}
        self.num_nodes_by_level = []
        self.nodes_by_level = []

        self.get_cv_metadata(cv)

    def get_cv_metadata(self, cv):
        '''
        Returns a node labelling scheme.
        Perform a per level labelling to each node.
        Defines the order in which each node appears in the matrix.
        '''

        level = 0
        q = deque([cv.root])
        visited = {}

        # Perform a per level traversal
        while q:
            level_size = len(q)

            curr_level = [] # nodes at the current level
            level_type = None
            for i in range(level_size):
                node = q.popleft()
                curr_level.append(node)

                if isinstance(node, Leaf):
                    continue
                else:
                    node_type = sum_type if isinstance(node, Sum) else prd_type
                    if level_type is None:
                        level_type = node_type
                    elif node_type != level_type:
                        error = "Level type mismatch: Expects " + level_type + " gets " + node_type
                        print(error)
                        raise Exception(error)


                for child in node.children:
                    if child in visited:
                        continue
                    visited[child] = True
                    q.append(child)

            self.type_by_level.append(level_type)
            self.nodes_by_level.append(curr_level)
            self.num_nodes_by_level.append(len(curr_level))
            for (label, node) in enumerate(curr_level):
                self.level_label_by_node[node] = label
            level += 1

        self.depth = level

        self.connections_by_level = self.get_connections_by_level(cv)
        self.masks_by_level = self.get_masks_by_level(cv)


    def get_connections_by_level(self, cv):
        '''
        Returns TorchSPN style sparse connection corresponding to ConvSPN cv
        '''
        connections_by_level = []
        for cur_level in range(self.depth - 1):
            next_level = cur_level + 1
            cur_level_count = self.num_nodes_by_level[cur_level]
            next_level_count = self.num_nodes_by_level[next_level]

            level_connections = defaultdict(list)

            cur_level_nodes = self.nodes_by_level[cur_level]
            for cur_node in cur_level_nodes:
                cur_label = self.level_label_by_node[cur_node]
                for child_node in cur_node.children:
                    child_label = self.level_label_by_node[child_node]
                    level_connections[cur_label].append(child_label)

            connections_by_level.append(level_connections)

        return connections_by_level

    def get_masks_by_level(self, cv):
        '''
        Returns a TorchSPN style matrix layer information corresponding to ConvSPN cv
        '''
        masks_by_level = []
        for cur_level in range(self.depth - 1):
            next_level = cur_level + 1
            cur_level_count = self.num_nodes_by_level[cur_level]
            next_level_count = self.num_nodes_by_level[next_level]

            level_mask = np.zeros((cur_level_count, next_level_count)).astype('float32')

            cur_level_nodes = self.nodes_by_level[cur_level]
            for cur_node in  cur_level_nodes:
                cur_label = self.level_label_by_node[cur_node]
                for child_node in cur_node.children:
                    child_label = self.level_label_by_node[child_node]
                    level_mask[cur_label][child_label] = 1

            level_mask = level_mask.T
            masks_by_level.append(level_mask)

        return masks_by_level

def get_edge_count_from_layers(cv_layers):
    '''
    Was used to test the correctness of edge count in the conversion
    '''

    edge_count = []
    for layer in cv_layers:
        edge_count.append(np.count_nonzero(layer))

    return edge_count
