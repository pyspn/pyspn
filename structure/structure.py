import numpy as np
import torch
import pdb
import math
from .leaf import *
from collections import defaultdict, deque

class Structure(object):
    def __init__(self):
        self.roots = []
        self.weights = []

    def print_level_stat(self, level, level_type, level_nodes, edge_count):
        print("Level " + str(level) + " (" + level_type + ") : " + str(len(level_nodes)) + " nodes, " + str(edge_count) + " edges")

    def print_level_weight_indices(self, level, level_type, level_nodes, edge_count):
        if level_type != "Sum":
            return

        str_by_nodes = []
        for node in level_nodes:
            node_idx = list(node.weight_id_by_child.values())
            str_by_nodes.append( str(node_idx) )

        level_node_str = ''.join(str_by_nodes)

        print(level_node_str)

    def print_stat(self):
        self.traverse_by_level(self.print_level_stat)

    def print_weight_indices(self):
        self.traverse_by_level(self.print_level_weight_indices)

    def traverse_by_level(self, fn):
        '''
        :param fn: takes in #level, level type, nodes, edge_count
        :return:
        '''
        q = deque(self.roots)

        level = 0
        total_nodes = 0
        total_edges = 0
        visited = {}

        while q:
            level_size = len(q)
            node_count = 0
            edge_count = 0

            level_type = None
            level_nodes = []
            while level_size:
                u = q.popleft()
                level_nodes.append(u)
                level_size -= 1

                level_type = u.node_type

                node_count += 1

                if isinstance(u,PixelLeaf):
                    continue

                edge_count += len(u.edges)

                for e in u.edges:
                    v = e.child
                    if v in visited:
                        continue

                    q.append(v)
                    visited[v] = True

            total_nodes += node_count
            total_edges += edge_count

            fn(level, level_type, level_nodes, edge_count)

            level += 1


        # Likelihoods is a dictionary of arrays, key of the dictionary is the id of the node. array is an array of numbers representing the likelihoods
        def parameter_update(root, likelihoods):
            root.count_n += 1
            if root.node_type == 'Product':
                for edge in root.edges:
                    parameter_update(edge.child, data)
            elif root.node_type == 'Sum':
                highest_child = []
                highest_child_likelihood = []
                highest_child_edge = []
                for edge in root.edges:
                    for idx, likelihood in enumerate(likelihoods[edge.child.id]):
                        if highest_child[idx] is None:
                            highest_child.append(edge.child)
                            highest_child_likelihood.append(likelihood)
                            highest_child_edge.append(edge)
                        elif likelihood > highest_child_likelihood[idx]:
                            higest_child[idx] = edge.child
                            highest_child_likelihood[idx] = likelihood
                            highest_child_edge[idx] = edge

                # TODO: prob can compress common highest childs
                for child in highest_child:
                    parameter_update(child, likelihoods)
                    highest_child_edge.count_n = (highest_child_edge.child.count_n + 1) / root.count_n

            # Assume Leaf otherwise
            else:
                # TODO: Update leaf counts once we get the structure right
                return

        def counting_parameter_update(likelihoods):
            for root in self.roots:
                parameter_update(root, likelihoods)


