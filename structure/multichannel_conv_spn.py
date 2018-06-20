class MultiChannelConvSPN(Structure):
    def __init__(self, x_size, y_size, sum_shifts, prd_subdivs, num_channels, num_out):
        super(MultiChannelConvSPN, self).__init__()

        self.x_size = x_size
        self.y_size = y_size
        self.sum_shifts = sum_shifts
        self.prd_subdivs = prd_subdivs
        self.num_channels = num_channels
        self.num_out = num_out

        self.weights = []

        self.cached_prd = defaultdict(list)
        self.cached_leaf = defaultdict(list)

        # statistics
        self.depth = 0
        self.total_sum = 0
        self.total_prd = 0
        self.reused_sum = 0
        self.reused_prd = 0
        self.count_by_depth = defaultdict(int)

        self.generate_spn()

    def get_ordered_leaves(self, leaves):
        '''
        Orders the leaves to match input
        [input_channel_0, ..., input_channel_(k-1)]

        Where input_channel is row-ordered
        '''
        print("Ordering Leaves...")
        leaves_input_indices = []
        per_network_size = self.x_size * self.y_size
        for leaf in leaves:
            index_within_network = int(leaf.y) * self.x_size + int(leaf.x)
            index = (leaf.network_id * per_network_size) + index_within_network

            leaves_input_indices.append(index)

        leaf_index_pairs = zip(leaves, leaves_input_indices)
        leaf_index_pairs = sorted(leaf_index_pairs, key=lambda pair: pair[1])

        sorted_leaves = [pair[0] for pair in leaf_index_pairs]

        return sorted_leaves

    def generate_spn(self):
        '''
        1. Generate root
        2. Generate #num_channels ConvSPN
        3. Connect separate ConvSPN's sum node
        '''
        root_scope = Scope(0, 0, self.x_size, self.y_size)
        self.roots = []
        for i in range(self.num_out):
            self.roots.append(Sum(root_scope))

        channels = []
        for i in range(self.num_channels):
            channel = ConvSPN(self.x_size, self.y_size, self.sum_shifts, self.prd_subdivs)

            for leaf_scope in channel.cached_leaf:
                leaf = channel.cached_leaf[leaf_scope]
                leaf.network_id = i

            self.populate_cache_from_spn(channel)
            channels.append(channel)

        # Update sum node weight indices to
        offset = 0
        for channel in channels:
            self.weights.extend( channel.weights )

            update_fn = (lambda level, level_type, level_nodes, edge_count: self.channel_level_update(
                offset, level, level_type, level_nodes, edge_count))
            channel.traverse_by_level(update_fn)

            offset += len(self.weights)

        self.weights = np.array(self.weights)

        # Now, cache contains prd and leaves from all channels.
        # So now we add inter-channel connections.
        for channel in channels:
            self.add_interchannel_connection(channel)

        # Set channels as root's child and update channel's depth.
        offset = len(self.weights)
        num_root_edge = 0
        for root in self.roots:
            channel_edges = channel.roots[0].edges

            edges = [] # all channel's root have identical children
            for e in channel_edges:
                weight_id =  num_root_edge + offset
                edge = SumEdge(parent=root, child=e.child, weight_id=weight_id)
                num_root_edge += 1
                edges.append(edge)

            root.edges = edges

        root_edge_weights = np.random.uniform(10, 1000, num_root_edge).astype('float32')

        self.weights = np.concatenate([ self.weights, root_edge_weights ])
        return

    def populate_cache_from_spn(self, spn):
        q = deque(spn.roots)
        visited = {}
        while q:
            level_size = len(q)

            while level_size:
                u = q.popleft()
                level_size -= 1

                if isinstance(u, Product):
                    self.cached_prd[u.scope.id].append(u)

                if isinstance(u, PixelLeaf):
                    self.cached_leaf[u.id].append(u)
                    continue

                for e in u.edges:
                    v = e.child
                    if v in visited:
                        continue

                    q.append(v)
                    visited[v] = True

    def add_interchannel_connection(self, spn):
        q = deque(spn.roots)
        visited = {}

        id_offset = len(self.weights)
        num_interchannel_edges = 0
        while q:
            level_size = len(q)

            while level_size:
                u = q.popleft()
                level_size -= 1

                if isinstance(u, PixelLeaf):
                    continue

                for e in u.edges:
                    v = e.child
                    if v in visited:
                        continue

                    q.append(v)
                    visited[v] = True

                if isinstance(u, Sum):
                    interchannel_children = []
                    old_children = [e.child for e in u.edges]
                    for v in old_children:
                        inter_child_in_v = self.cached_prd[v.scope.id]
                        interchannel_children.extend(inter_child_in_v)

                    new_children = list( set(interchannel_children) - set(old_children) )
                    for c in new_children:
                        weight_id = id_offset + num_interchannel_edges
                        edge = SumEdge(parent=u, child=c, weight_id=weight_id)
                        num_interchannel_edges += 1

                        u.edges.append(edge)

        interchannel_weights = np.random.uniform(10, 1000, num_interchannel_edges).astype('float32')
        self.weights = np.concatenate( [self.weights, interchannel_weights] )

    def channel_level_update(self, offset, level, level_type, level_nodes, edge_count):
        if level_type != "Sum":
            return

        for node in level_nodes:
            for e in node.edges:
                e.weight_id += offset
