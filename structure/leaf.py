class Leaf(object):
    def __init__(self, id):
        self.id = id

class PixelLeaf(Leaf):
    def __init__(self, x, y):
        id = str([x, y])
        super(PixelLeaf, self).__init__(id)
        self.x = x
        self.y = y
        self.depth = 0
        self.node_type = "Leaf"
        self.network_id = None

class BinaryLeaf(Leaf):
    def __init__(self, id):
        super(BinaryLeaf, self).__init__(id)
