import numpy
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

mod = SourceModule("""
__global__ void add(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] + b[i];
}
""")
add_gpu = mod.get_function("add")

NUM = 32

# generate two array on host
x = numpy.ones(NUM).astype(numpy.float32)
y = numpy.ones(NUM).astype(numpy.float32)

# copy data from host to GPU and allocate memory for result
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)
res_gpu = gpuarray.zeros((NUM, 1), numpy.float32)

# apply "GPU kernel"
add_gpu(res_gpu, x_gpu, y_gpu, block=(NUM,1,1), grid=(1,1))

# copy result from device to host
res_cpu = res_gpu.get()

# sanity check
print("Result: " + str(res_cpu.T))
