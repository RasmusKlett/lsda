#run restoreImage first

import numpy as np
import Image

pixels=np.loadtxt("../../data/image.csv",delimiter=",")
pixels=pixels*255/8
im=Image.fromarray(pixels)
im.show()

