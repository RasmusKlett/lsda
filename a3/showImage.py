#run restoreImage first

from pylab import *

pixels=np.loadtxt("../../data/image.csv",delimiter=",")/8

imshow(pixels)
show()
