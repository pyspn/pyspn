class FlatSPN(object):
    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size

        self.generate_spn()

    def generate_spn(self):
        root_scope = Scope(0, 0, self.x_size, self.y_size)
        self.roots = [Sum(root_scope)]

        for y in range(self.y_size):
            for x in range(self.x_size):
                child_leaf = PixelLeaf(x, y)
                self.roots[0].children.append(child_leaf)
