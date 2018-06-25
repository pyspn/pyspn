class Leaf(object):
    def __init__(self, id, var_num=None):
        self.id = id
        self.var_num = var_num

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

class TachyonBinaryLeaf(Leaf):
    def __init__(self, id, var_num, w1, w2):
        super(TachyonBinaryLeaf, self).__init__(id, var_num)
        self.w1 = w1
        self.w2 = w2
        self.node_type = "TachyonBinaryLeaf"
