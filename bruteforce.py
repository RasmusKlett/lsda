import numpy, time
docword=numpy.loadtxt("data/docword.kos.txt",skiprows=3)

def makeMatrix():

	start=time.time()
	M=max(docword[:,0])
	N=max(docword[:,1])
	JacMat=numpy.zeros((M,N))
	
	for ind in range(len(docword)):
		i=docword[ind,0]
		j=docword[ind,1]
		JacMat[i-1,j-1]=1
	Corr=numpy.dot(JacMat,numpy.transpose(JacMat))
	sim=numpy.zeros((len(Corr),len(Corr)))
	for i in range(len(Corr)):
		for j in range(i+1,len(Corr)):
			sim[i,j]=Corr[i,j]/(Corr[i,i]+Corr[j,j]-Corr[i,j])
	
	end=time.time()
	t=end-start
	print t
	return sum(sum(sim))/(M*(M-1)/2)
