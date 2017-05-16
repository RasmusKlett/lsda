import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import ras_task_1_4 as ras


model=pickle.load(open("../../data/pickleTree.p", "rb"))

lines_pr_chunk=30000
chunk_size=lines_pr_chunk*9
chunk=np.empty((0,chunk_size))
i=0
image=np.zeros((3000,3000))


with open("../../data/landsat_set.csv", "r") as in_file:
    lines=0
    for line in in_file:
        
        ll=np.fromstring(line,dtype=float,sep=",")
        chunk=np.append(chunk,ll)

        lines=lines+1
        if lines==lines_pr_chunk:
            chunk.resize((lines_pr_chunk,9))


            pred=ras.predict(chunk)
            
            for j,pixel in enumerate(pred[0,:]):
            
                ii=j%3000
                jj=int(np.floor((j-ii)/3000))
                image[ii,jj+i]=pixel
            i+=10
            lines=0
             
            chunk=np.empty((0,chunk_size))
            
            print(i)
          
            
np.savetxt("../../data/image.csv",image,delimiter=",")



