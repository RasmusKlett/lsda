from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
from pickle import dump, load

tree_args = {
"n_estimators": 10,
"criterion":"gini",
"max_depth":None,
"min_samples_split":2,
"max_features":None,
}

dataset=np.loadtxt("../../data/landsat_train_subset.csv",delimiter=",",dtype=int)
y=dataset[:,0];
X=dataset[:,1:]

model=RFC(**tree_args)

m=model.fit(X,y,sample_weight=None)

with open("../../data/pickleTree.p", "wb") as picle_file:
    dump(model, picle_file)
print("Training accuracy: ", model.score(X,y,sample_weight=None))


