import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle


model=pickle.load(open("../../data/pickleTree.p", "rb"))

lines_pr_chunk=3000
chunk_size=lines_pr_chunk*9
chunk=np.empty((0,chunk_size))
i=0
image=np.zeros((3000,3000))


with open("../../data/landsat_set.csv", "r") as in_file:

    for line in in_file:
        ll=np.fromstring(line,dtype=float,sep=",")
        chunk=np.append(chunk,ll)


        if np.size(chunk,0)>=chunk_size:
            chunk.resize((lines_pr_chunk,9))


            pred=model.predict(chunk)
            for j,pixel in enumerate(pred):
                image[i,j]=pixel
            i+=1
            
             
            chunk=np.empty((0,chunk_size))
            
            print(i)
          
            
np.savetxt("../../data/image.csv",image,delimiter=",")



