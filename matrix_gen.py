import numpy as np
import torch
import pdb
import math
from collections import defaultdict, deque
from struct_gen import *

# Constants
sum_type = "sum"
prd_type = "prd"
class Layer(object):
    def __init__(self):
        self.type = None
        self.weights = []
        self.mask = []

class CVMetaData(object):
    def __init__(self, cv):
        self.depth = 0
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

        # Perform a per level traversal
        while q:
            level_size = len(q)

            curr_level = [] # nodes at the current level
            visited = {}
            for i in range(level_size):
                node = q.popleft()
                curr_level.append(node)

                if isinstance(node, Leaf):
                    continue

                for child in node.children:
                    if child in visited:
                        continue
                    visited[child] = True
                    q.append(child)

            self.nodes_by_level.append(curr_level)
            self.num_nodes_by_level.append(len(curr_level))
            for (label, node) in enumerate(curr_level):
                self.level_label_by_node[node] = label
            level += 1

        self.depth = level

def get_layers(cv):
    '''
    Returns a TorchSPN style layer information corresponding to ConvSPN cv
    '''

    metadata = CVMetaData(cv)
    masks_by_level = []
    for cur_level in range(metadata.depth - 1):
        next_level = cur_level + 1
        cur_level_count = metadata.num_nodes_by_level[cur_level]
        next_level_count = metadata.num_nodes_by_level[next_level]

        level_mask = np.zeros((cur_level_count, next_level_count))

        cur_level_nodes = metadata.nodes_by_level[cur_level]
        for cur_node in  cur_level_nodes:
            cur_label = metadata.level_label_by_node[cur_node]
            for child_node in cur_node.children:
                child_label = metadata.level_label_by_node[child_node]
                level_mask[cur_label][child_label] = 1

        masks_by_level.append(level_mask)

    return masks_by_level

cv = ConvSPN(32, 32, 8, 2)
cv.generate_spn()

cv.print_stat()

pdb.set_trace()
