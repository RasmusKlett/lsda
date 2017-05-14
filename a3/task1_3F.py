import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

model=pickle.load(open("../../data/pickleTree.p", "rb"))

lines_pr_chunk=10000
chunk_size=lines_pr_chunk*10
chunk=np.empty((0,chunk_size))
i=0
score=0


with open("../../data/landsat_train_remaining.csv", "r") as in_file:
    for line in in_file:
        ll=np.fromstring(line,dtype=float,sep=",")
        chunk=np.append(chunk,ll)


        if np.size(chunk,0)>=chunk_size:
            chunk.resize((lines_pr_chunk,10))

            y=chunk[:,0]
            X=chunk[:,1:]
            score=np.add(score,sum(model.predict(X)==y)/len(y))

            chunk=[]
            i+=1
            print(i)
            
if np.size(chunk,0)<chunk_size:
    n=len(chunk)
    chunk.resize((n/10,10))
    y=chunk[:,0]
    X=chunk[:,1:]
    score=np.add(score,sum(model.predict(X)==y)/len(y))
print score/i

