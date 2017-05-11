import numpy as np
from itertools import groupby, combinations

NUM_DOCS = 3430
NUM_WORDS = 6906

kos = np.loadtxt("docword.kos.txt", skiprows=3, usecols=(0, 1), dtype=int)

docs = []

# Convert into a tuple per document containing id and array of word ids
for doc_id, rows in groupby(kos, lambda row: row[0]):
    docs.append(
        (doc_id, np.fromiter((row[1] for row in rows), dtype=int))
    )


def calcJaccard(words1, words2):
    union = float(np.union1d(words1, words2).size)
    intersection = float(np.intersect1d(words1, words2).size)
    return intersection / union

# For each combination of tupes calculate Jaccard similarity
jaccards = np.fromiter(
    (calcJaccard(words1, words2)
        for (_, words1), (_, words2)
        in combinations(docs, 2)),
    dtype=np.float64,
)

print("Mean of Jaccard similarities")
print(np.mean(jaccards))
