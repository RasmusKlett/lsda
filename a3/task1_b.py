import numpy as np

subset = np.random.choice(25667779, 100000, replace=False)
subset.sort()
print(subset[1:20])
position = 0
with open("../../data/landsat_train.csv", "r") as in_file:
    with open("../../data/landsat_train_subset.csv", "w") as out_file:
        with open("../../data/landsat_train_remaining.csv", "w") as remaining_file:
            for idx, line in enumerate(in_file):
                if position < subset.size and subset[position] == idx:
                    out_file.write(line)
                    # print(idx, line[1:10])
                    position = position + 1
                else:
                    remaining_file.write(line)
