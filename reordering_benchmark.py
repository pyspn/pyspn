import numpy as np
import torch
import pdb
import math
from collections import defaultdict, deque
import os.path
import sys
from struct_gen import *
from matrix_gen import *
from struct_to_spn import *
import math
from random import randint
import time
from torch.autograd import Variable as Variable
import cProfile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from TorchSPN.src import network, param, nodes
from reordering.masks import reorder

debug = True
start = None
end = None
pr = None

def profile_start():
    global start
    start = time.time()

def profile_end():
    global start
    global end

    end = time.time()

    duration = end - start
    print("Duration: " + str(duration))

    return duration

def cprofile_start():
    global pr
    pr = cProfile.Profile()
    pr.enable()

def cprofile_end():
    global pr
    pr.disable()
    pr.dump_stats('rb.cprof')

def classic_pass(mask, weight, input):
    batch = input.size()[0]
    num = len(mask)

    maxval = torch.max( input )
    maxval.detach() # disconnect during bp. any constant works here

    tmp = input - maxval # log(x/max)

    tmp = torch.exp(tmp) # x/max
    trueweight = mask * weight
    val = torch.mm(tmp, trueweight) # <w, x>/max

    val += torch.exp(torch.FloatTensor([-75]))[0]
    val = torch.log(val)
    val += maxval

    return val

def get_torch_masks_and_weights(masks):
    torch_masks = []
    torch_weights = []

    for np_mask in masks:
        np_weights = np.copy(np_mask)
        mask = Variable( torch.from_numpy(np_mask) )
        mask.detach()

        weights = torch.nn.Parameter( torch.from_numpy(np_weights), requires_grad=True)

        torch_masks.append(mask)
        torch_weights.append(weights)

    return (torch_masks, torch_weights)

def torch_sparse_pass(indices, sparse_weights, input):
    cprofile_start()
    num_out = len(indices)

    val = torch.zeros(num_out)
    for (i, index) in enumerate(indices):
         val[i] = torch.dot(input[index], sparse_weights[i])

    cprofile_end()

    return val

def get_sparse_idx_wgt(mask, weights):
    indices = []
    sparse_weights = []

    num_cols = len(mask[0])
    num_rows = len(mask)

    for c in range(num_cols):
        col_idx = []
        col_wg = []
        for r in range(num_rows):
            if mask[r][c]:
                col_idx.append(r)
                col_wg.append(weights[r][c])
        col_idx =  torch.tensor(col_idx)
        col_wg = torch.FloatTensor(col_wg)

        indices.append(col_idx)
        sparse_weights.append(col_wg)

    return (indices, sparse_weights)

def bmm_pass(mask, weight, input):
    batch = input.size()[0]
    num = len(mask)

    maxval = torch.max( input )
    maxval.detach() # disconnect during bp. any constant works here

    tmp = input - maxval # log(x/max)

    tmp = torch.exp(tmp) # x/max
    trueweight = mask * weight
    val = torch.bmm(tmp, trueweight) # <w, x>/max

    val += torch.exp(torch.FloatTensor([-75]))[0]
    val = torch.log(val)
    val += maxval

    return val

def test_forward(masks, weights):
    batch = 1
    num_layers = len(masks)

    for i in reversed(range(num_layers)):
        input = torch.ones(batch, len(masks[i]))
        mask = masks[i]
        weight = weights[i]
        classic_pass(mask, weight, input) # TODO: Cross-layer pass

def collapse_matrix(matrix):
    stat = reorder.get_stat(matrix)

    diag_matrix = stat[1]
    boxes = stat[2]
    if not stat[4]:
        print("ERROR: unequal block size")
        return

    num_boxes = len(boxes)
    box_dim = stat[3][0]

    collapsed_matrix = np.zeros((num_boxes, box_dim[0], box_dim[1]), dtype='float32')

    for (i, box) in enumerate(boxes):
        r_start, r_end, c_start, c_end= box[0], box[1], box[2], box[3]
        submatrix = diag_matrix[r_start:r_end+1, c_start:c_end+1]
        collapsed_matrix[i] = submatrix

    batch = 1
    input = Variable( torch.ones(num_boxes, batch, box_dim[0]), requires_grad=False)

    mask = Variable( torch.from_numpy(collapsed_matrix), requires_grad=False )
    weight = torch.nn.Parameter( torch.from_numpy(collapsed_matrix), requires_grad=True)

    return (mask, weight, input)

def quick_sparse_pass(mask, weight, dim, input):
    sel = input[:,mask]
    dot_res = torch.mul(sel, weight)
    condensed = torch.reshape(dot_res, dim )

    return torch.sum(condensed, 1)

def get_sparse_mwd(mask, weight):
    (idx, wgt) = get_sparse_idx_wgt(mask, weight)

    max_length = max([len(a) for a in idx])
    new_idx = []
    new_weight = []
    for ii in range(len(idx)):
        id = idx[ii]
        new_idx.extend(id)

        buffer = [new_idx[-1]] * (max_length - len(id))
        new_idx.extend(buffer)

        wg = wgt[ii]
        new_weight.extend(wg)

        buffer_wg = [0] * (max_length - len(wg))
        new_weight.extend(buffer_wg)

    new_idx = torch.tensor(new_idx)
    new_weight = torch.FloatTensor(new_weight)
    part_lg = len(idx)

    return (new_idx, new_weight, (part_lg, max_length) )

def test_speedup():
    structure = MultiChannelConvSPN(16, 16, 1, 2, 10)
    shared_parameters = param.Param()
    network = MatrixSPN(
        structure,
        shared_parameters,
        is_cuda=False)

    masks = network.masks

    print("Mask generated")

    (torch_masks, torch_weights) = get_torch_masks_and_weights(masks)

    collapsed_data = []
    mask_stats = []
    for (i, mask) in enumerate(masks):
        print("Collapsing " + str(i))
        x = collapse_matrix(mask)
        collapsed_data.append(x)
        mask_stats.append(reorder.get_stat(mask))
        break

    print("Collapsed!")

    sparse_tuple = []
    for mask in masks:
        (idx, wgt) = get_sparse_idx_wgt(mask, mask)
        input = torch.ones(len(mask))
        sparse_tuple.append((idx, wgt, input))

    speedups = []
    normal_total = 0
    speedup_total = 0
    # num_masks = len(masks) # TODO: Enable this again
    num_masks = 1

    for i in range(num_masks):
        num_iters = 10

        # Sparse
        (idx, wgt, ipt) = sparse_tuple[i]
        print("NO IDX " + str(len(idx)))

        # '''
        # Approach 1
        # '''
        # profile_start()
        # for it in range(num_iters):
        #     num_out = len(idx)
        #
        #     val = torch.zeros(num_out)
        #     for (j, index) in enumerate(idx):
        #         id_ipt = ipt[index]
        #         w = wgt[j]
        #         val[j] = torch.dot(id_ipt, w)
        # profile_end()

        # '''
        # Approach 2
        # '''
        # max_length = max([len(a) for a in idx])
        # new_idx = []
        # new_weight = []
        # for ii in range(len(idx)):
        #     id = idx[ii]
        #     new_idx.extend(id)
        #
        #     buffer = [new_idx[-1]] * (max_length - len(id))
        #     new_idx.extend(buffer)
        #
        #     wg = wgt[ii]
        #     new_weight.extend(wg)
        #
        #     buffer_wg = [0] * (max_length - len(wg))
        #     new_weight.extend(buffer_wg)
        #
        # new_idx = torch.tensor(new_idx)
        # new_weight = torch.FloatTensor(new_weight)
        # part_lg = len(idx)
        #
        # profile_start()
        # for it in range(num_iters):
        #     sel = ipt[new_idx]
        #     multipl = torch.mul(sel, new_weight)
        #     rearr = torch.reshape(multipl, (part_lg, max_length) )
        #     v = torch.sum(rearr, 1)
        #
        # profile_end()

        # Collapsed tensor
        (tmask, tweights, input) = collapsed_data[i]

        profile_start()
        for it in range(num_iters):
            bmm_pass(tmask, tweights, input)
        sp_dur = profile_end()
        speedup_total += sp_dur

        batch = 1
        mask = torch_masks[i]
        input = torch.ones(batch, len(mask))
        weight = torch_weights[i]
        '''
        Approach 2 (condensed)
        '''
        (m, w, d) = get_sparse_mwd(torch_masks[i], torch_weights[i])
        profile_start()
        for it in range(num_iters):
            val = quick_sparse_pass(m, w, d, input)
        profile_end()

        '''
        Classic
        '''
        profile_start()
        for it in range(num_iters):
             classic_pass(mask, weight, input)
        normal_dur = profile_end()
        normal_total += normal_dur
        #
        # p_speedup = normal_dur / sp_dur

        # speedups.append(p_speedup)
    #
    # print(speedups)
    # print("Normal " + str(normal_total))
    # print("Speedup " + str(speedup_total))

    # reordered_mask_stats = []
    # for mask in network.masks:
    #     stat = reorder.get_stat(mask)
    #     reordered_mask_stats.append(stat)

def test_swap():
    structure = MultiChannelConvSPN(16, 16, 1, 2, 10)
    shared_parameters = param.Param()
    network = MatrixSPN(
        structure,
        shared_parameters,
        is_cuda=False)

    masks = network.masks
    # mask = masks[0]
    # for mask in masks:
    #     print("Mask: " + str(mask.shape))

    prev_mask = masks[0]
    next_mask = masks[1]

    (remask, col_swaps) = reorder.get_reordered_matrix(prev_mask)
    (remask_double_T, row_swaps) = reorder.get_reordered_matrix(remask.T)
    remask_double = remask_double_T.T

    print("Done")

def test_correctness():
    '''
    Test the correctness of implementation of matrix reordering
    '''

    structure = MultiChannelConvSPN(32, 32, 1, 2, 10)
    shared_parameters = param.Param()
    network = MatrixSPN(
        structure,
        shared_parameters,
        is_cuda=False)

    masks = network.masks

    for mask in masks:
        print(mask.shape)

    mask = masks[0]
    weight = np.multiply(mask, np.random.uniform(-100, 100, mask.shape))

    (reweight, col_swaps) = reorder.get_reordered_matrix(weight)
    (reweight_double_T, row_swaps) = reorder.get_reordered_matrix(reweight.T)
    reordered_weight = reweight_double_T.T
    # reordered_weight = reweight

    old_weight = torch.FloatTensor( weight )

    new_weight = torch.FloatTensor( reordered_weight )

    input_arr = np.array(list(range(len(mask))))
    swapped_input = input_arr[row_swaps]

    input = torch.FloatTensor( [input_arr] )
    swapped_input = torch.FloatTensor( [swapped_input] )

    res_1 = torch.mm(input, old_weight)[0][col_swaps]
    res_2 = torch.mm(swapped_input,new_weight)[0]

    diff = torch.max(torch.abs(res_1 - res_2))

    print("Done " + str(diff))

    return

def get_is_bd(mask):
    decomp = reorder.disjoint_decomposition(mask)
    return reorder.is_block_diagonal(decomp)

def get_plot(mask):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    plt.imshow(mask)
    plt.show()

def test_consecutive():
    '''
    Test the correctness of implementation of matrix reordering
    '''

    structure = MultiChannelConvSPN(4, 4, 4, 2, 3)
    shared_parameters = param.Param()
    network = MatrixSPN(
        structure,
        shared_parameters,
        is_cuda=False)

    masks = network.masks

    for mask in masks:
        print(mask.shape)

    mask = masks[0]

    (row_mask_T, row_swaps) = reorder.get_reordered_matrix(mask.T)
    row_mask = row_mask_T.T

    (re_mask, col_swaps) = reorder.get_reordered_matrix(row_mask)

    decomp = reorder.disjoint_decomposition(re_mask)
    is_bd = reorder.is_block_diagonal(decomp)

    masks[0] = re_mask

    reorder_maps = []

    if is_bd:
        print("IS BD 0")

    for i in range(1, len(masks)):
        mi = masks[i]
        mi = mi[col_swaps]

        (row_mask_T, row_swaps) = reorder.get_reordered_matrix(mi.T)
        row_mask = row_mask_T.T

        reorder_maps.append(row_swaps)

        (re_mi, cs_i) = reorder.get_reordered_matrix(row_mask)

        bd = get_is_bd(re_mi)

        if not bd:
            print("Not BD " + str(i))
            pdb.set_trace()
            return
        else:
            print("IS BD " + str(i))
            pdb.set_trace()

        col_swaps = cs_i
        masks[i] = re_mi

    return

def main():
    # test_consecutive()
    test_speedup()
    # test_swap()
    # structure = MultiChannelConvSPN(16, 16, 1, 2, 10)
    # shared_parameters = param.Param()
    # network = MatrixSPN(
    #     structure,
    #     shared_parameters,
    #     is_cuda=False)

if __name__=='__main__':
    main()
