import numpy as np

subset_size = 1000000

subset = np.random.choice(25667779, subset_size, replace=False)
subset.sort()
position = 0
with open("../../data/landsat_train.csv", "r") as in_file:
    with open("../../data/landsat_large_subset.csv", "w") as out_file:
        for idx, line in enumerate(in_file):
            if position < subset.size and subset[position] == idx:
                out_file.write(line)
                position = position + 1
