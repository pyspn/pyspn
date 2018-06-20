class Scope(object):
    def __init__(self, x_ori, y_ori, x_size, y_size):
        self.x_ori = x_ori
        self.y_ori = y_ori
        self.x_size = x_size
        self.y_size = y_size

        self.x_end = x_ori + x_size - 1
        self.y_end = y_ori + y_size - 1

        total_coord = [x_ori, y_ori, x_size, y_size]
        self.id = str(x_ori) + "_" + str(y_ori) + "_" + \
            str(x_size) + "_" + str(y_size)
