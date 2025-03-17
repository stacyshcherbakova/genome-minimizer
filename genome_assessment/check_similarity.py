## checks if you have any identical sequences in your npy file 

import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.directories import *

arrays = np.load(DATA_DIR+"cleaned_genes_lists.npy", allow_pickle=True)

n = len(arrays)
identical_pairs = []

for i in range(n):
    # print(i)
    for j in range(i + 1, n):
        if np.array_equal(arrays[i], arrays[j]):
            identical_pairs.append((i, j))

if identical_pairs:
    print("Identical arrays found at the following indices:")
    for pair in identical_pairs:
        print(f"Array {pair[0]} and Array {pair[1]} are identical.")
else:
    print("No identical arrays found.")