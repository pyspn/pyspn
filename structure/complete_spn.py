class CompleteSPN(Structure):
    def __init__(self, num_variables, sum_factor, prd_factor):
        super(CompleteSPN, self).__init__()

        self.num_variables = num_variables
        self.sum_factor = sum_factor
        self.prd_factor = prd_factor

        self.roots = None
        self.leaves = []
        self.generate_structure()

    def generate_structure(self):
        self.roots = [self.generate_sum(0, self.num_variables - 1)]

    def generate_sum(self, start, end):
        scope_size = end - start + 1

        # If scope only contain a PixelLeaf, replace with a PixelLeaf node
        if scope_size == 1:
            leaf = PixelLeaf(0, 0)
            leaf.id = start
            leaf.network_id = 0
            return leaf

        # If scope contains multiple leaves, create a sum_node
        node = Sum()

        # Sum always have #sum_factor children
        for i in range(self.sum_factor):
            # Scope of its children are identical
            child_prd = self.generate_prd(start, end)
            node.children.append(child_prd)

        return node

    def generate_prd(self, start, end):
        node = Product()

        scope_size = end - start + 1

        # If we have less than #prd_factor leaves in our scope, then just
        # branch on whatever's remaining
        num_branches = min(scope_size, self.prd_factor)

        full_child_scope_size = math.ceil(scope_size / num_branches)
        remainder_child_scope_size = scope_size % full_child_scope_size

        # Generate the remainder children
        if remainder_child_scope_size > 0:
            child_start = end - remainder_child_scope_size + 1
            child_end = end
            child_sum = self.generate_sum(child_start, child_end)
            node.children.append(child_sum)

            # since we've generated the remainder, we have 1 less branch to generate
            num_branches -= 1

        # Generate the first #(num_branches - 1) children
        for i in range(num_branches):
            child_start = start + i * full_child_scope_size
            child_end = child_start + full_child_scope_size - 1
            child_sum = self.generate_sum(child_start, child_end)

            node.children.append(child_sum)

        return node

    def get_ordered_leaves(self, leaves):
        return leaves
