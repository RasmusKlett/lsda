import numpy as np
from sklearn.ensemble import RandomForestClassifier
from math import ceil
from pickle import dump, load
from scipy import stats


num_trees = 100
trees_per_forest = 1

run_count = int(ceil(float(num_trees) / trees_per_forest))

pickle_dir = "../../data/huge_forest"


def get_large_dataset():
    return np.loadtxt(
        "../../data/landsat_large_subset.csv",
        delimiter=",",
        dtype=int
    )


def build_forest(dataset):
    for i in range(run_count):
        print("Building forest " + str(i))

        forest = RandomForestClassifier(
            n_estimators=trees_per_forest,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            max_features=None,
        )
        X = dataset[:, 1:]
        Y = dataset[:, 0]

        forest.fit(X, Y)

        with open(pickle_dir + "/tree" + str(i) + ".p", "wb") as picle_file:
            dump(forest, picle_file)


def predict(dataset):
    predictions = []

    for i in range(run_count):
        print("Evaluating forest " + str(i))

        with open(pickle_dir + "/tree" + str(i) + ".p", "rb") as picle_file:
            forest = load(picle_file)

        predictions.append(forest.predict(dataset))

    return stats.mode(np.array(predictions))[0]


if __name__ == "__main__":
    dataset = get_large_dataset()
    build_forest(dataset)
    prediction = predict(dataset[:, 1:])
    is_right = prediction == dataset[:, 0]

    print("Prediction on training data", prediction)
    print("Accuracy", float(np.sum(is_right)) / dataset.shape[0])
