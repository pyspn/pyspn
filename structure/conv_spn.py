class ConvSPN(GraphSPN):
    def __init__(self, x_size, y_size, sum_shifts, prd_subdivs):
        super(ConvSPN, self).__init__()

        self.x_size = x_size
        self.y_size = y_size
        self.sum_shifts = sum_shifts
        self.prd_subdivs = prd_subdivs

        self.weights = []

        self.cached_sum = {}
        self.cached_prd = {}
        self.cached_leaf = {}

        self.last_weight_id = -1

        # statistics
        self.depth = 0
        self.total_sum = 0
        self.total_prd = 0
        self.reused_sum = 0
        self.reused_prd = 0
        self.count_by_depth = defaultdict(int)

        self.generate_spn()

    def generate_spn(self):
        root_scope = Scope(0, 0, self.x_size, self.y_size)
        self.roots = [self.generate_sum(root_scope, 0)]

        num_weights = self.last_weight_id + 1
        self.weights = np.random.uniform(10, 1000, num_weights).astype('float32')

    def generate_leaf(self, scope, depth):
        if scope.id in self.cached_leaf:
            return self.cached_leaf[scope.id]

        leaf = PixelLeaf(scope.x_ori, scope.y_ori)
        leaf.depth = depth
        self.cached_leaf[scope.id] = leaf

        return leaf

    def generate_sum(self, scope, depth):
        self.total_sum += 1
        self.depth = max(self.depth, depth)
        child_depth = depth + 1

        # Check if we have a PixelLeaf node
        if scope.x_size == 1 and scope.y_size == 1:
            return self.generate_leaf(scope, child_depth)

        # Return cached sum if available
        cached_sum = self.get_cached_sum(scope)
        if cached_sum:
            self.reused_sum += 1
            return cached_sum

        # update statistics
        self.count_by_depth[depth] += 1

        # Generate root and all of its child products from shifts
        root = Sum(scope=scope)
        root.id = scope.id
        root.depth = depth

        shifts = self.generate_shifts(scope)

        for shift in shifts:
            prd = self.generate_prd(shift, child_depth)
            self.last_weight_id += 1

            edge = SumEdge(parent=root, child=prd, weight_id=self.last_weight_id)
            root.edges.append(edge)

        # Cache root
        self.cached_sum[scope.id] = root

        return root

    def generate_prd(self, scope, depth):
        self.total_prd += 1
        self.depth = max(self.depth, depth)
        child_depth = depth + 1

        # Return cached prd if available
        cached_prd = self.get_cached_prd(scope)
        if cached_prd:
            self.reused_prd += 1
            return cached_prd

        # update statistics
        self.count_by_depth[depth] += 1

        # Generate root and all of its child sums from subdivisions
        root = Product(scope=scope)
        root.id = scope.id
        root.depth = depth

        subdivs = self.generate_subdivisions(scope)

        for subdiv in subdivs:
            sum = self.generate_sum(subdiv, child_depth)

            edge = ProductEdge(parent=root, child=sum)

            root.edges.append(edge)

        # Cache root
        self.cached_prd[scope.id] = root

        return root

    def get_cached_sum(self, scope):
        if scope.id in self.cached_sum:
            return self.cached_sum[scope.id]

        return None

    def get_cached_prd(self, scope):
        if scope.id in self.cached_prd:
            return self.cached_prd[scope.id]

        return None

    def generate_shifts(self, scope):
        # sum_shifts
        '''
        To determine:
        - what is a reasonable shift?
        '''

        x_stride = math.ceil(float(scope.x_size) / float(self.sum_shifts))
        y_stride = math.ceil(float(scope.y_size) / float(self.sum_shifts))

        if x_stride == 0 or y_stride == 0:
            return []

        x_max = int(min(self.sum_shifts, scope.x_size))
        y_max = int(min(self.sum_shifts, scope.y_size))
        x_offsets = [i * x_stride for i in range(x_max)]
        y_offsets = [i * y_stride for i in range(y_max)]

        shifts = []
        for x_offset in x_offsets:
            for y_offset in y_offsets:
                shift = Scope(
                    scope.x_ori + x_offset,
                    scope.y_ori + y_offset,
                    scope.x_size,
                    scope.y_size)
                shifts.append(shift)

                '''
                Clips scope to parent's scope.
                The assumption wrt size ordering would be messed up, so matrix generation
                doesn't work as neat as we'd like it to be. Figure this out soon.
                '''
                # x_ori = scope.x_ori + x_offset
                # y_ori = scope.y_ori + y_offset
                #
                # # ensure that the end doesn't get out of the scope
                # x_end = min(scope.x_end, x_ori + scope.x_size - 1)
                # y_end = min(scope.y_end, y_ori + scope.y_size - 1)
                #
                # x_size = x_end - x_ori + 1
                # y_size = y_end - y_ori + 1
                #
                # shift = Scope(x_ori, y_ori, x_size, y_size)

        return shifts

    def scope_is_out_of_bounds(self, scope):
        # Get coordinates of all 4 corners
        a = (scope.x_ori, scope.y_ori)
        b = (scope.x_end, scope.y_ori)
        c = (scope.x_end, scope.y_end)
        d = (scope.x_ori, scope.y_end)

        corners = [a, b, c, d]

        # If any of the corner is inside the box, then the scope isn't out of bound
        for (x, y) in corners:
            if 0 <= x < self.x_size and 0 <= y < self.y_size:
                return False

        return True

    def generate_subdivisions(self, scope):
        '''
        for each subdivs
            pass if it's completely OOB, generate otherwise.
        '''

        x_size = math.ceil(float(scope.x_size) / float(self.prd_subdivs))
        y_size = math.ceil(float(scope.y_size) / float(self.prd_subdivs))

        if x_size == 0 or y_size == 0:
            return []

        x_max = int(min(self.prd_subdivs, scope.x_size))
        y_max = int(min(self.prd_subdivs, scope.y_size))
        x_offsets = [i * x_size for i in range(x_max)]
        y_offsets = [i * y_size for i in range(y_max)]

        subdivs = []
        for x_offset in x_offsets:
            for y_offset in y_offsets:
                subdiv = Scope(
                    scope.x_ori + x_offset,
                    scope.y_ori + y_offset,
                    x_size,
                    y_size)

                if self.scope_is_out_of_bounds(subdiv):
                    continue

                subdivs.append(subdiv)

        return subdivs

    def naive_traverse_by_level(self):
        '''
        Traverse the SPN as if subtree sharing isn't implemented. This would take too long on large SPNs.
        '''
        q = deque(self.roots)

        level = 0
        while q:
            level_size = len(q)
            all = []
            while level_size:
                u = q.popleft()
                level_size -= 1

                all.append(u.id)

                if isinstance(u,PixelLeaf):
                    continue
                for e in u.edges:
                    v = e.child
                    q.append(v)

            print("Level " + str(level) + ": " + str(len(all)))
            level += 1
