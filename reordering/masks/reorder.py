import numpy as np
import torch
import pdb
import math
from collections import defaultdict, deque
import os.path
import sys
from numpy import genfromtxt

def get_reordered_matrix(mask):
    mask = np.copy(mask)
    anchor = -1
    num_rows = len(mask)
    num_cols = len(mask[0])

    indices = list(range(num_cols))

    for row_idx in range(num_rows):
        row = mask[row_idx]
        swap_cnt = 0
        for col_idx in range(num_cols):
            if col_idx <= anchor:
                continue

            if row[col_idx] > 0:
                swap_cnt += 1
                anchor += 1
                # swap
                mask[:,[anchor, col_idx]] = mask[:,[col_idx, anchor]]
                indices[anchor], indices[col_idx] = indices[col_idx], indices[anchor]

    return (mask, indices)

def disjoint_decomposition(matrix):
    mask = np.copy(matrix)
    num_rows = len(mask)
    num_cols = len(mask[0])

    sizes = []

    def fill(row_idx, col_idx, mask):
        min_r, max_r, min_c, max_c = row_idx, row_idx, col_idx, col_idx

        if mask[row_idx][col_idx] == 0:
            return None

        q = deque([(row_idx, col_idx)])

        while q:
            (row_idx, col_idx) = q.pop()

            if mask[row_idx][col_idx] == 0:
                continue

            min_r = min(row_idx, min_r)
            max_r = max(row_idx, max_r)
            min_c = min(col_idx, min_c)
            max_c = max(col_idx, max_c)
            mask[row_idx][col_idx] = 0

            if 0 < row_idx: # up
                up = (row_idx - 1, col_idx)
                q.append(up)

            if row_idx < num_rows - 1: # down
                down = (row_idx + 1, col_idx)
                q.append(down)

            if 0 < col_idx: # left
                left = (row_idx, col_idx - 1)
                q.append(left)

            if col_idx < num_cols - 1: # right
                right = (row_idx, col_idx + 1)
                q.append(right)

        return (min_r, max_r, min_c, max_c)

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            if mask[row_idx][col_idx] > 0:
                new_size = fill(row_idx, col_idx, mask)
                sizes.append(new_size)

    return sizes

def get_dimensions(sizes):
    dims = []
    for size in sizes:
        dims.append((size[1] - size[0] + 1, size[3] - size[2] + 1))
    return dims

def is_dim_eq(dims):
    x, y = None, None
    for dim in dims:
        if x is None:
            x = dim[0]
            y = dim[1]
        else:
            if x != dim[0] or y != dim[1]:
                return False
    return True

def is_block_diagonal(decomp):
    def is_overlapping(rect_1, rect_2):
        if rect_1[0] <= rect_2[0] and rect_2[1] <= rect_1[1]:
            return True
        if rect_1[2] <= rect_2[2] and rect_2[3] <= rect_1[3]:
            return True

        return False

    for (i, rect_1) in enumerate(decomp):
        for (j, rect_2) in enumerate(decomp):
            if i != j:
                overlaps = is_overlapping(rect_1, rect_2)
                if overlaps:
                    print("Overlap " + str(rect_1) + " " + str(rect_2))
                    return False

    return True

def get_stat(mask):
    (remask, col_swaps) = get_reordered_matrix(mask)
    (remask_double, row_swaps) = get_reordered_matrix(remask.T)
    remask_double = remask_double.T
    decomp = disjoint_decomposition(remask_double)
    dims = get_dimensions(decomp)
    is_eq = is_dim_eq(dims)
    block_diag = is_block_diagonal(decomp)

    return (remask, remask_double, decomp, dims, is_eq, block_diag)

# def main():
#     print("Loading matrices...")
#
#     # mask_name = 'mask_1_prd'
#     # mask = genfromtxt(mask_name, delimiter=',')
#     # x = get_stat(mask)
#     #
#     # print("Done")
#
# if __name__=='__main__':
#     main()
