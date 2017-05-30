import time
import numpy
from sklearn.ensemble import ExtraTreesClassifier

data = numpy.genfromtxt("landsat_train.csv", delimiter=",")
X, y = data[:,1:], data[:,0]
print("Training data: n=%i, d=%i" % (X.shape[0], X.shape[1]))

n_jobs=1
print("Using n_jobs=%i" % n_jobs)
model = ExtraTreesClassifier(n_estimators=36, 
                             criterion='gini',
                             max_depth=10, 
                             min_samples_split=50, 
                             max_features="sqrt",
                             n_jobs=n_jobs)
start_time = time.time()
model.fit(X, y)
end_time = time.time()
print("Fitting time: %f" % (end_time - start_time))
