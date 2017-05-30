# NOTE: Use your the virtual machine to execute this code and
#       keep the memory allocated for the machine to 2GB!

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import time
import matplotlib.pyplot as plt

# load data
print("Loading training data ...")
data_train = np.genfromtxt("../../data/hw5/neighbors/train.csv", comments="#", delimiter=",")
Xtrain, ytrain = data_train[:,:-1], data_train[:,-1]
print("Loaded training data: n=%i, d=%i" % (Xtrain.shape[0], Xtrain.shape[1]))


print("Loading validation data ...")
data_val = np.genfromtxt("../../data/hw5/neighbors/validation.csv", comments="#", delimiter=",")
Xval, yval = data_val[:,:-1], data_val[:,-1]
print("Loaded validation data: n=%i, d=%i" % (Xval.shape[0], Xval.shape[1]))

MSE=[]
total_features=Xval.shape[1]
total_instances=Xval.shape[0]
n_features=5
untouched=range(15)
picked=[]
print("Feature selection...")
for i in range(n_features):
    error=[]
    
    for j in untouched:
        feat=picked+[j]
        print(j)
        Xtrain_tmp=Xtrain[:,feat] 
        model = KNeighborsRegressor(n_neighbors=10, algorithm="kd_tree")
        model.fit(Xtrain_tmp, ytrain)
        diff=np.subtract(model.predict(Xval[:,feat]),yval)
        error.append(np.sqrt(np.dot(diff,diff)))
        

    winner = untouched[np.argmin(error)]
    untouched=np.delete(untouched,np.argmin(error))
    picked.append(winner)
    MSE.append(np.min(error)/total_instances)
    num=i+1
    print("%i features chosen: " % num, picked)
    print("Validation error: ", np.min(error)/total_instances )
    


Xtrain=Xtrain[:,picked]
model = KNeighborsRegressor(n_neighbors=10, algorithm="kd_tree")
model.fit(Xtrain_tmp, ytrain)
print("Model fitted!")

# testing phase (apply model to a big test set!)

start=time.time()
print("Loading testing data ...")
data_test = np.genfromtxt("../../data/hw5/neighbors/test.csv", comments="#", delimiter=",")
Xtest, ytest = data_test[:,picked], data_test[:,-1]
print("Loaded testing data: n=%i, d=%i" % (Xtest.shape[0], Xtest.shape[1]))

print("Applying model ...")
# FIXME: Fitting the model on all test points causes a MemoryError! 
preds = model.predict(Xtest)

extime=time.time()-start


print("Predictions computed for %i patterns ...!" % len(preds))
print("Mean of predictions: %f" % np.mean(np.array(preds)))
print("Execution time of testing phase: %f" % extime)

diff=np.subtract(preds,ytest)
MSEtest=np.sqrt(np.dot(diff,diff))/Xtest.shape[0]
print("MSE on test data: ", MSEtest)

plt.plot(range(1,6),MSE)
plt.ylabel("MSE")
plt.title("MSE during feature selection")
plt.xlabel("Number of features")
plt.show()


