import numpy as np
import torch
import pdb
import math
from collections import defaultdict, deque
from struct_to_spn import *

x_size = 32
y_size = 32
mspn = MatrixSPN(x_size, y_size, 8, 2)

# fake_input = np.random.rand(x_size * y_size)
fake_input = np.zeros(x_size * y_size)
mspn.feed(fake_input)

prob = mspn.forward()
print(prob)

pdb.set_trace()
