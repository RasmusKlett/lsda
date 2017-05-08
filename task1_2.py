import numpy
import time

docword = numpy.loadtxt("data/docword.kos.txt",skiprows=3,dtype="int")
# docword=numpy.array([[1,1,1],[1,2,1],[1,3,1],[2,3,1],[2,4,1],[2,5,1]])

print("Creating identity matrix")
M = int(max(docword[:, 0]))
N = int(max(docword[:, 1]))
print(M)
charac = numpy.zeros((M, N))

for ind in range(len(docword)):
    i = docword[ind, 0]
    j = docword[ind, 1]
    charac[i-1, j-1] = 1


def hashVal(a, x, b):
    h = (a * x + b) % (N + 1)
    if h == N:
        h -= h
    return h


def computeSig(K):
    print("Computing minHash-matrix")
    signature = numpy.zeros((M, K))
    start=time.time()
    for k in range(K):
        a = numpy.random.randint(1, 100) + 1
        b = numpy.random.randint(1, 100)
        for docId in range(M):
            jac = 0
            tries = 0
            while jac == 0:
                jac = charac[docId, hashVal(a, tries, b)]
                tries += 1
            signature[docId, k] = tries
        print(k)
    end=time.time()
    t=end-start
    print(t)
    return signature


def computeSimilarities(signatures, K):
    print("Comparing signatures")
    similarities = numpy.zeros((M, M))
    for i in range(M):
        for j in range(i+1, M):
            similarities[i, j] = numpy.sum(signatures[i, :] == signatures[j, :])
    similarities = similarities / K
    return similarities




if __name__ == "__main__":
    K = 100
    Sig = computeSig(K)
    sim = computeSimilarities(Sig, K)
    print(sim)
    print(sum(sum(sim))/(M*(M-1)/2))
