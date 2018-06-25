import numpy as np
from .structure import Structure
from collections import defaultdict, deque
from .scope import Scope
from .node import *
from .leaf import *
import math

class FileToSPN(Structure):
    def __init__(self, filePath):
        super(FileToSPN, self).__init__()
        self.generate_spn(filePath)

    def generate_spn(filePath):
        lines = open(filepath).read().splitlines()

        currentlyReading = ''
        nodesById = {}
        nodesByIdWithoutParents = {}

        weights = []

        for line in lines:
            if line == '##NODES##':
                currentlyReading = 'nodes'
            elif line == '##EDGES##':
                currentlyReading = 'edges'
            else:
                values = line.split(',')
                print(line)

                if currentlyReading == 'nodes':
                    if values[1] == 'SUM':
                        entry = {'nodeType': 'sum'}
                        entry['node'] = Sum()
                        entry['node'].id = int(values[0])

                        pprint(vars(entry['node']))
                        nodesById[int(values[0])] = entry
                        nodesByIdWithoutParents[int(values[0])] = entry['node']

                    elif values[1] == 'PRD':
                        entry = {'nodeType': 'product'}
                        entry['node'] = Product()
                        entry['node'].id = int(values[0])

                        pprint(vars(entry['node']))
                        nodesById[int(values[0])] = entry
                        nodesByIdWithoutParents[int(values[0])] = entry['node']

                    elif values[1] == 'BINNODE':
                        entry = {'nodeType': 'binnode'}
                        entry['node'] = TachyonBinaryLeaf(int(values[0]), int(values[2]), float(values[3]), float(values[4]))

                        nodesById[int(values[0])] = entry
                        nodesByIdWithoutParents[int(values[0])] = entry

                        pprint(vars(entry['node']))

                elif currentlyReading == 'edges':
                    #values[0] is parent id
                    #values[1] is child id
                    #values[2] is weight
                    if nodesById[int(values[0])]['nodeType'] == 'sum':
                        parentNode = nodesById[int(values[0])]['node']
                        childNode = nodesById[int(values[1])]['node']
                        edge = SumEdge(parentNode, childNode)

                        weight_id = len(weights)
                        weights.append(float(values[2]))
                        edge.weight_id = weight_id

                        if nodesByIdWithoutParents[int(values[1])]:
                            print('nodeswithoutparents')
                            print(int(values[1]))
                            del nodesByIdWithoutParents[int(values[1])]


                        pprint(vars(edge))

                    elif nodesById[int(values[0])]['nodeType'] == 'product':
                        parentNode = nodesById[int(values[0])]['node']
                        childNode = nodesById[int(values[1])]['node']
                        edge = ProductEdge(parentNode, childNode)

                        if nodesByIdWithoutParents[int(values[1])]:
                            del nodesByIdWithoutParents[int(values[1])]
                        pprint(vars(edge))

                    print('edge')

        # Roots:
        roots = []
        for id, node in nodesByIdWithoutParents.items():
            roots.append(node)

        self.weights = weights
        self.roots = roots
