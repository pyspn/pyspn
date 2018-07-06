class Node(object):
    def __init__(self, scope, node_type=None):
        self.edges = []
        self.scope = scope
        self.depth = 0
        self.id = None
        self.node_type = node_type
        self.count_n = 1

class Edge(object):
    def __init__(self, parent=None, child=None):
        self.parent = parent
        self.child = child
        self.count_n = 1

class SumEdge(Edge):
    def __init__(self, parent=None, child=None, weight_id=None):
        Edge.__init__(self, parent=parent, child=child)
        self.weight_id = weight_id

class ProductEdge(Edge):
    def __init__(self, parent=None, child=None):
        Edge.__init__(self, parent=parent, child=child)

class Sum(Node):
    def __init__(self, scope=None):
        Node.__init__(self, scope, node_type="Sum")

class Product(Node):
    def __init__(self, scope=None):
        Node.__init__(self, scope, node_type="Product")
