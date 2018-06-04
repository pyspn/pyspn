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
from torch import optim

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from TorchSPN.src import network, param, nodes
from reordering.masks import reorder
from torch import cuda

debug = True
start = None
end = None
pr = None

is_cuda = cuda.is_available()
def var(tensor):
    if is_cuda:
        tensor = tensor.cuda()

    return Variable(tensor, requires_grad=False)

def parameter(tensor):
    if is_cuda:
        tensor = tensor.cuda()

    return torch.nn.Parameter(tensor, requires_grad=True)

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
    profile_start()
    pr = cProfile.Profile()
    pr.enable()

def cprofile_end():
    global pr
    pr.disable()
    pr.dump_stats('rb.cprof')
    return profile_end()

def classic_pass(mask, weight, input):
    batch = input.size()[0]

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

def natural_sparse_pass(weight, input):
    batch = input.size()[0]

    maxval = torch.max( input )
    maxval.detach() # disconnect during bp. any constant works here

    tmp = input - maxval # log(x/max)

    tmp = torch.exp(tmp) # x/max
    val = torch.mm(tmp, weight) # <w, x>/max

    val += torch.exp(torch.FloatTensor([-75]))[0]
    val = torch.log(val)
    val += maxval

    return val

def get_torch_masks_and_weights(masks):
    torch_masks = []
    torch_weights = []

    for np_mask in masks:
        np_mask = np.array(np_mask, dtype='float32')
        np_weights = np.copy(np_mask)
        mask = var( torch.from_numpy(np_mask) )
        mask.detach()

        weights = parameter( torch.from_numpy(np_weights) )

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
            if (mask[r, c].data != 0).cpu().numpy():
                col_idx.append(r)
                col_wg.append(weights[r, c].data.cpu())

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
    input = var( torch.ones(num_boxes, batch, box_dim[0]))

    mask = var( torch.from_numpy(collapsed_matrix) )
    weight = parameter( torch.from_numpy(collapsed_matrix) )

    return (mask, weight, input)

def quick_sparse_pass(mask, weight, dim, input):
    val = None

    maxval = torch.max(input, 1)[0]
    maxval.detach()

    batch = len(input)
    maxval = maxval.view(batch, 1)

    tmp = input - maxval
    tmp_exp = torch.exp(tmp)
    long = tmp_exp[:, mask]
    # long = tmp_exp[:, mask]
    dot_res = torch.mul(long, weight)
    condensed = dot_res.view(batch, dim[0], dim[1])
    result = torch.sum(condensed, 2)

    return result

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

    new_idx = torch.LongTensor(new_idx)
    new_weight = torch.from_numpy(np.array(new_weight, dtype='float32'))
    part_lg = len(idx)

    return (var(new_idx), parameter(new_weight), (part_lg, max_length) )

def get_sparse_mtx(mask, weights):
    num_rows = len(mask)
    num_cols = len(mask[0])

    sz = mask.size()
    mask = mask.data.cpu().numpy()
    weights = weights.data.cpu().numpy()

    sparse_idx = []
    sparse_weights = []
    for r in range(num_rows):
        for c in range(num_cols):
            if mask[r, c]:
                sparse_idx.append( [r, c] )
                sparse_weights.append(weights[r, c])

    sparse_idx = torch.LongTensor(sparse_idx)
    sparse_weights = torch.from_numpy(np.array(sparse_weights))

    sparse_mtx = torch.sparse.FloatTensor(
        sparse_idx.t(), sparse_weights, sz)

    return sparse_mtx

def nop(mask, weight):
    pass

def test_speedup():
    structure = MultiChannelConvSPN(8, 8, 4, 2, 8)
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

    shared_parameters = param.Param()
    mwd = []
    sms = []
    for i, mask in enumerate(masks):
        sm = get_sparse_mtx(torch_masks[i], torch_weights[i])
        sms.append(sm)

        (m, w, d) = get_sparse_mwd(torch_masks[i], torch_weights[i])
        mwd.append((m,w,d))

        shared_parameters.add_param(w, hook=None)

    # speedups = []
    new_sparse_total = 0
    normal_total = 0
    speedup_total = 0
    num_masks = len(masks) # TODO: Enable this again

    # opt = optim.Adam( shared_parameters.para_list, lr=.03, weight_decay=0.01)

    for i in range(num_masks):
        num_iters = 100

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
        # (tmask, tweights, input) = collapsed_data[i]
        #
        # profile_start()
        # for it in range(num_iters):
        #     bmm_pass(tmask, tweights, input)
        # sp_dur = profile_end()
        # speedup_total += sp_dur

        batch = 10
        mask = torch_masks[i]
        input = var(torch.ones(batch, len(mask)))
        weight = torch_weights[i]

        '''
        Approach 3 (PyTorch Sparse)
        '''
        sparse_weight = sms[i]
        cprofile_start()
        for it in range(num_iters):
            val = natural_sparse_pass(weight, input)
        new_sp_dur = cprofile_end()
        new_sparse_total += new_sp_dur

        '''
        Approach 2 (condensed)
        '''
        (m, w, d) = mwd[i]
        cprofile_start()
        for it in range(num_iters):
            val = quick_sparse_pass(m, w, d, input)
            # loss = torch.sum(val)
            # loss.backward()
            # if it % 10 == 0:
                # print("VAL " + str(val))
            # opt.step()

        sp_dur = cprofile_end()
        speedup_total += sp_dur

        # '''
        # Classic
        # '''
        profile_start()
        for it in range(num_iters):
            val = classic_pass(mask, weight, input)

        normal_dur = profile_end()
        normal_total += normal_dur

        # p_speedup = normal_dur / sp_dur

        # speedups.append(p_speedup)
    #
    # print(speedups)
    print("Normal " + str(normal_total))
    print("New sparse " + str(new_sparse_total))
    print("Speedup " + str(speedup_total))

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
