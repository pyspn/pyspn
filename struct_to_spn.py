import numpy as np
import torch
import pdb
import math
from collections import defaultdict, deque
from struct_gen import *
from matrix_gen import *

cv = ConvSPN(32, 32, 8, 2)
cv.print_stat()

metadata = CVMetaData(cv)

# self.depth = 0
# self.masks_by_level = []
# self.type_by_level = []

print(metadata.depth)
print(metadata.masks_by_level)
print(metadata.type_by_level)
