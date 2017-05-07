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

# For each combination of tuples compute Jaccard similarity
jaccards = []
for (doc_id1, words1), (doc_id2, words2) in combinations(docs, 2):
    union = float(np.union1d(words1, words2).size)
    intersection = float(np.intersect1d(words1, words2).size)
    jaccards.append(
        (doc_id1, doc_id2, intersection / union)
    )

print("Mean of Jaccard similarities")
print(np.mean(list(item[2] for item in jaccards)))
