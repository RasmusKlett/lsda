# https://github.com/gieseke/bufferkdtree/tree/master/examples
import time
from bufferkdtree import NearestNeighbors
import generate

Xtrain, Ytrain, Xtest = generate.get_data_set(
            data_set="psf_model_mag", NUM_TRAIN=1000000, 
            NUM_TEST=1000000)

n_jobs = 1
print("Using n_jobs=%i" % n_jobs)
nbrs = NearestNeighbors(n_neighbors=10, algorithm="kd_tree", 
            leaf_size=32, n_jobs=n_jobs)
nbrs.fit(Xtrain)

start_time = time.time()
_, _ = nbrs.kneighbors(Xtest)
end_time = time.time()
print("Testing time: %f" % (end_time - start_time))
