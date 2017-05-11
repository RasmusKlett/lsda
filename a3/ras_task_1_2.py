import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pickle import dump

landsat_subset = np.loadtxt(
    "../../data/landsat_train_subset.csv",
    delimiter=",",
    dtype=int
)


forest = RandomForestClassifier(
    n_estimators=10,
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    max_features=None,
)
X = landsat_subset[:, 1:-1]
Y = landsat_subset[:, 0]

forest.fit(X, Y)

training_score = forest.score(X, Y)
print("Training set score", training_score)


with open("../../data/pickleTree.p", "wb") as picle_file:
    dump(forest, picle_file)
