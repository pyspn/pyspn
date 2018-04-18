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
        self.level_label_by_node = {}
        self.num_nodes_by_level = []

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

            self.num_nodes_by_level.append(len(curr_level))
            for (label, node) in enumerate(curr_level):
                self.level_label_by_node[node] = label
            level += 1

# def get_layers(cv):
#     '''
#     Returns a TorchSPN style layer information corresponding to ConvSPN cv
#     '''
#
#     level = 0
#     q = deque([cv.root])
#
#     # Perform a per level traversal
#     while q:
#         level_size = len(q)
#
#         curr_level = [] # nodes at the current level
#         visited = {} # ensures dedup
#         for i in range(level_size):
#             node = q.popleft()
#             curr_level.append(node)
#             for child in node.children:
#                 if child in visited:
#                     continue
#                 visited[child] = True
#                 q.append(child)
#
#         level += 1



cv = ConvSPN(32, 32, 8, 2)
cv.generate_spn()

cv.print_stat()

cv_metadata = CVMetaData(cv)
print(cv_metadata.num_nodes_by_level)
print(cv_metadata.level_label_by_node)
