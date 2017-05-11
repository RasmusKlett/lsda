import numpy
import time

docword = numpy.loadtxt("docword.kos.txt", skiprows=3, dtype=int)


def makeMatrix():

    start = time.time()
    M = int(max(docword[:, 0]))
    N = int(max(docword[:, 1]))
    JacMat = numpy.zeros((M, N))

    for ind in range(len(docword)):
        i = docword[ind, 0]
        j = docword[ind, 1]
        JacMat[i-1, j-1] = 1

    Corr = numpy.dot(JacMat, numpy.transpose(JacMat))
    sim = numpy.zeros((M, M))

    for i in range(M):
        for j in range(i+1, M):
            sim[i, j] = float(Corr[i, j]) / float(Corr[i, i]+Corr[j, j]-Corr[i, j])

    end = time.time()
    t = end-start
    print(t)
    numpy.savetxt("bruteforce_sim.txt",sim)
    avg= float(sum(sum(sim)))/float(M*(M-1)/2)
    print(avg)


makeMatrix()
