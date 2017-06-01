import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import matplotlib.pyplot as plt
import time
import math



###x = numpy.linspace(0, 2*numpy.pi, 10)
#Prepared data
x = numpy.linspace(-10, 10, 10000000)
y = numpy.arctan(x)


#######################################
############## CUDA ###################
#######################################
x = x.astype(numpy.float32)
y = y.astype(numpy.float32)

#Create interval for interpolation
##interval = numpy.asarray([0, 2*numpy.pi])
interval = numpy.asarray([-10, 10])
min_el = numpy.min(interval)
max_el = numpy.max(interval)

#To transfer data to allocate memory on the device
x_gpu = cuda.mem_alloc(x.nbytes)
y_gpu = cuda.mem_alloc(y.nbytes)

#To transfer the data to the GPU
cuda.memcpy_htod(x_gpu, x)
cuda.memcpy_htod(y_gpu, y)

#Kernel module on C 
mod = SourceModule("""
  #include <stdio.h>
  __global__ void linear_interpolation(float *x, float *y, const float min_el, const float max_el)
  {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int jdy = threadIdx.y + blockDim.y*blockIdx.y;
    x[idx] = y[idx]-x[idx]/(max_el-min_el)*1+x[idx];
  }
  """)

#To get the kernel function
func = mod.get_function("linear_interpolation")
start_timer_1 = time.time()
func(x_gpu, y_gpu, numpy.float32(min_el), numpy.float32(max_el), block=(1000,1,1))
x_int = numpy.empty_like(x)
cuda.memcpy_dtoh(x_int, x_gpu)

stop_timer_1 = time.time()
timer_1 = stop_timer_1 - start_timer_1
print('CUDA: ', timer_1, 'sec.')


########################################
######## NUMPY INTERPOLATION ###########
########################################

xvals = numpy.linspace(0, 2*numpy.pi, 10000000)
start_timer_2 = time.time()
yinterp = numpy.interp(xvals,x,y)
stop_timer_2 = time.time()
timer_2 = stop_timer_2 - start_timer_2
print('NUMPY: ', timer_2, 'sec.')

########################################
####### BASE PYTHON INTERPOLATION ######
########################################

start_timer_3 = time.time()
x_list = numpy.array(x).tolist()
y_list = []
interval_list = [-10, 10]
x_iterp = []
for i in x_list:
    y_element = math.atan(i)
    y_list.append(y_element)

for j in range(len(x_list)):
    x_int_elem = (y_list[j] - x_list[j])/(interval_list[1] - interval_list[0])*1 + x_list[j]
    x_iterp.append(x_int_elem)
stop_timer_3 = time.time()
timer_3 = stop_timer_3 - start_timer_3
print('PYTHON BASE: ', timer_3, 'sec.')

#######################################
############### END ###################
#######################################


#x_l = [1, 2, 3]
#x_label = ['CUDA', 'NUMPY', 'BASE PYTHON']
#y_label = [timer_1, timer_2, timer_3]
#plt.axis([1, 3, 0., 1])
#plt.grid(True)
#plt.bar(x_l,y_label, align='center')
#plt.ylabel('Time, sec.')
#plt.title('Comparison CUDA vs Python and Numpy')
#plt.xticks(x_l, x_label)
#plt.show()
