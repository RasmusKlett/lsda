import time
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np
import skcuda.linalg as culinalg
import skcuda.misc as cumisc
culinalg.init()

n, m, k = 20000, 10000, 20000

# matrices on host
A = np.asarray(np.random.rand(n, m), np.float32)
B = np.asarray(np.random.rand(m, k), np.float32)
C = np.asarray(np.random.rand(k, n), np.float32)

# CPU
print("Starting computations on CPU ...")
time_start = time.time()
res_host = np.dot(np.dot(A, B), C)
time_end = time.time()
host_time = time_end - time_start

# matrices on device
print("Copying data from main memory to GPU memory ...")
time_start = time.time()
A_gpu = gpuarray.to_gpu(A)
B_gpu = gpuarray.to_gpu(B)
C_gpu = gpuarray.to_gpu(C)
time_end = time.time()
transfer_time = time_end - time_start

# GPU
print("Starting computations on GPU ...")
time_start = time.time()
res_gpu = culinalg.dot(A_gpu, B_gpu)
res_gpu = culinalg.dot(res_gpu, C_gpu)
time_end = time.time()
gpu_time = time_end - time_start

# Sanity check
print("Equal: " + str(np.allclose(res_host, res_gpu.get())))

# Runtime Comparison
print("CPU:\t\t%f" % host_time)
print("Transfer:\t%f" % transfer_time)
print("GPU:\t\t%f" % gpu_time)
print("Speed-Up (with transfer):\t%f" % (host_time/(gpu_time+transfer_time)))
print("Speed-Up (without transfer):\t%f" % (host_time/gpu_time)

