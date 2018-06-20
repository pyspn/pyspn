class ClassifierSPN(GraphSPN):
    def __init__(self, x_size, y_size, sum_shifts, prd_subdivs, num_channels, num_classes):
        super(ClassifierSPN, self).__init__()

        self.x_size = x_size
        self.y_size = y_size
        self.sum_shifts = sum_shifts
        self.prd_subdivs = prd_subdivs
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.generate_spn()

    def generate_spn(self):
        root_scope = Scope(0, 0, self.x_size, self.y_size)
        root = Sum(root_scope)

        classes = []
        for c in range(num_classes):
            classifier = MultiChannelConvSPN(self.x_size, self.y_size, self.sum_shifts, self.prd_subdivs, self.num_channels)
            class_root = self.create_class_node(classifier, c)
            classes.append(class_root)

        root.children = classes

    def create_class_node(self, classifier, class_id):
        '''
        Create a product node with the class and its one-hot
        '''

        # note that it actually has an additional binary variable
        class_scope = Scope(0, 0, self.x_size, self.y_size)
        class_root = Product(class_scope)

        classifier_depth = classifier.depth

        binary_root = BinaryLeaf(class_id)
        binary_depth = 0
        empty_scope = Scope(0, 0, 0, 0)
        while binary_depth < classifier_depth:
            if binary_depth % 2 == 0:
                new_root = Product(empty_scope)
            else:
                new_root = Sum(empty_scope)

            new_root.children = [binary_root]
            binary_root = new_root

            binary_depth += 1

        class_root.children = [binary_root, classifier.root]

        return class_root
