def overlap(scope1, scope2):
    '''
    Two scopes overlap if they have some common elements between them
    '''
    scope1 = list(map(int, scope1.split('_')))
    scope2 = list(map(int, scope2.split('_')))
    x_ori_1 = scope1[0]  # xo1
    x_end_1 = scope1[2]  # xe1
    y_ori_1 = scope1[1]  # yo1
    y_end_1 = scope1[3]  # ye1
    x_ori_2 = scope2[0]  # xo2
    x_end_2 = scope2[2]  # xe2
    y_ori_2 = scope2[1]  # yo2
    y_end_2 = scope2[3]  # ye2
    # Case 1: (xo1 <= xo2 <= xe1) && (yo1 <= yo2 <= ye1)
    if ((x_ori_1 <= x_ori_2 <= x_end_1) and (y_ori_1 <= y_ori_2 <= y_end_1)) or ((x_ori_2 <= x_ori_1 <= x_end_2) and (y_ori_2 <= y_ori_1 <= y_end_2)):
        return True
    # Case 2: (xo1 <= xe2 <= xe1) && (yo1 <= y02 <= ye1)
    elif ((x_ori_1 <= x_end_2 <= x_end_1) and (y_ori_1 <= y_ori_2 <= y_end_1)) or ((x_ori_2 <= x_end_1 <= x_end_2) and (y_ori_2 <= y_ori_1 <= y_end_2)):
        return True
    # Case 3: (xo1 <= xo2 <= xe1) && (yo1 <= ye2 <= ye1)
    elif ((x_ori_1 <= x_ori_2 <= x_end_1) and (y_ori_1 <= y_end_2 <= y_end_1)) or ((x_ori_2 <= x_ori_1 <= x_end_2) and (y_ori_2 <= y_end_1 <= y_end_2)):
        return True
    # Case 4: (xo1 <= xe2 <= xe1) && (yo1 <= ye2 <= ye1)
    elif ((x_ori_1 <= x_end_2 <= x_end_1) and (y_ori_1 <= y_end_2 <= y_end_1)) or ((x_ori_2 <= x_end_1 <= x_end_2) and (y_ori_2 <= y_end_1 <= y_end_2)):
        return True
    else:
        return False


def check_validity(root):
    '''
    checks the consistency and completeness of the SPN
    returns a tuple of id and True/False
    '''
    scope_hash = set()
    for child in root.children:
        if isinstance(child, PixelLeaf):
            if len(scope_hash) == 0:
                scope_hash.add(child.id)
            if root.node_type == 'Sum':
                if child.id not in scope_hash:
                    return -1, False
                else:
                    # prod
                    if child.id in scope_hash:
                        pdb.set_trace()
                        return -1, False
                    else:
                        scope_hash.add(child.id)
        else:
            child_id, valid_flag = check_validity(child)
            if valid_flag:
                if len(scope_hash) == 0:
                    scope_hash.add(child_id)
                if root.node_type == 'Sum':
                    if child_id not in scope_hash:
                        return -1, False
                else:
                    for scope in scope_hash:
                        if overlap(child_id, scope):
                            pdb.set_trace()
                            return -1, False
                        else:
                            scope_hash.add(child_id)
            else:
                return (-1, False)
    return root.id, True
