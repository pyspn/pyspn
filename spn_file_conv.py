import pdb
from collections import defaultdict, deque
from struct_gen import *

class TachyonFileGenerator(object):
    def __init__(self, cv):
        self.text =  self.generate_text(cv)

    def write_to_file(self, filename):
        file = open(filename,'w')
        file.write(self.text)

    def generate_text(self, cv):
        (flattened_spn, labels_by_node) = self.flatten_spn(cv)

        print("Nodes: " + str(len(flattened_spn)))

        # pdb.set_trace()

        return self.generate_node_text(flattened_spn, labels_by_node) + \
            self.generate_edges_text(flattened_spn, labels_by_node)

    def flatten_spn(self, cv):
        q = deque([cv.root])

        flattened_spn = []
        labels_by_node = {}
        visited = {}

        label = 0
        while q:
            level_size = len(q)
            print(level_size)

            while level_size:
                u = q.popleft()

                flattened_spn.append(u)
                labels_by_node[u] = label

                level_size -= 1
                label += 1

                if isinstance(u, Leaf):
                    continue

                for v in u.children:
                    if v in visited:
                        continue

                    q.append(v)
                    visited[v] = True

        return (flattened_spn, labels_by_node)

    def generate_node_text(self, flattened_spn, labels_by_node):
        # add nodes header
        all_nodes_text = []
        header = '####NODES####\n'
        all_nodes_text.append(header)

        for node in flattened_spn:
            id = labels_by_node[node]
            node_descriptor = None
            if isinstance(node, Sum):
                node_descriptor = str(id) + ',SUM\n'
            elif isinstance(node, Product):
                node_descriptor = str(id) + ',PRD\n'
            else:
                prob = 0.6
                prob_c = 1 - prob
                node_descriptor = str(id) + ',BINNODE,' + str(id) + ',' \
                    + str(prob) + ',' + str(prob_c) + '\n'

            all_nodes_text.append(node_descriptor)

        nodes_text = ''.join(all_nodes_text)

        return nodes_text

    def generate_edges_text(self, flattened_spn, labels_by_node):
        # add edges header
        all_edges_text = []
        header = '####EDGES####\n'
        all_edges_text.append(header)

        for node in flattened_spn:
            if isinstance(node, Sum):
                id = labels_by_node[node]
                for child in node.children:
                    child_id = labels_by_node[child]
                    weight = 0.5
                    edge_descriptor = str(id) + ',' + str(child_id) + ',' + str(weight) + '\n'
                    all_edges_text.append(edge_descriptor)
            elif isinstance(node, Product):
                id = labels_by_node[node]
                for child in node.children:
                    child_id = labels_by_node[child]
                    edge_descriptor = str(id) + ',' + str(child_id) + '\n'
                    all_edges_text.append(edge_descriptor)

        edges_text = ''.join(all_edges_text)

        return edges_text

cv = ConvSPN(32, 32, 8, 2)
cv.print_stat()

tf = TachyonFileGenerator(cv)
tf.write_to_file('cv.spn')

# pdb.set_trace()
