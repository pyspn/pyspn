import numpy as np
import torch
import pdb
import math
from collections import defaultdict, deque
from struct_gen import ConvSPN

cv = ConvSPN(32, 32, 8, 2)
cv.generate_spn()

cv.print_stat()
