from cp_utils import decompose_QR_KERNELS
import serial_utils as SU
from cp_utils import init_tensor
from numpy import random
import numpy as np
import cupy as cp
import time


uniform_test= True
tall_skinny_test = False
cpu_test = False
num_tests = 1
max_iterations = 1


print("INITIALIZING BENCMARK")

if uniform_test:
    print("RANDOM DIMENSIONS",flush=True)
    for test in range(0,num_tests):


        rank = random.randint(200,300)
        dim1 = random.randint(200,300)
        dim2 = random.randint(200,300)
        dim3 = random.randint(200,300)
        #dim1 = 300
      #  dim2 = 300
      #  dim3 = 300
      #  rank = 300

        a_cp = cp.random.random((dim1,rank),dtype=cp.float32)
        b_cp = cp.random.random((dim2,rank),dtype=cp.float32)
        c_cp = cp.random.random((dim3,rank),dtype=cp.float32)

        #a_np = cp.asnumpy(a_cp.copy())
        #b_np = cp.asnumpy(b_cp.copy())
        #c_np = cp.asnumpy(c_cp.copy())

        A1,A2,A3,X = init_tensor(dim1,dim2,dim3,rank)
        #np.save('Benchmark_files/RegularFactorMatrices_a.npy', a_np)
        #np.save('Benchmark_files/RegularFactorMatrices_b.npy', b_np)
        #np.save('Benchmark_files/RegularFactorMatrices_c.npy', c_np)

        ## Time CUDA
        start_time_cuda = time.time()
        S = decompose_QR_KERNELS(X,rank=rank,max_iter=max_iterations,factor_matrices=[a_cp,b_cp,c_cp],verbose=False)
        end_time_cuda = time.time()

        start_time_cuda_cutensor = time.time()
        S = decompose_QR_KERNELS(X,rank=rank,max_iter=max_iterations,factor_matrices=[a_cp,b_cp,c_cp],verbose=False,cutensor_ttm=True)
        end_time_cuda_cutensor = time.time()

        #if cpu_test:
        #    X = cp.asnumpy(X.copy())
        #    start_time_serial = time.time()
        #    S = SU.decompose_QR_SERIAL(X,rank=rank,factor_matrices=[a_np,b_np,c_np],max_iter=max_iterations,verbose=False)
        #    end_time_serial = time.time()

        print("TEST NUMBER: ",test)
        print("Rank: ",rank)
        print("Dim1: ",dim1)
        print("Dim2: ",dim2)
        print("Dim3: ",dim3)
        print("Cuda Time: ",end_time_cuda-start_time_cuda)
        print("Cutensor Time: ",end_time_cuda_cutensor-start_time_cuda_cutensor)

if tall_skinny_test:
    print("\nTALL SKINNY DIMENSIONS",flush=True)
    for test in range(num_tests):


        rank = random.randint(10,20)
        dim1 = random.randint(200,300)
        dim2 = random.randint(200,300)
        dim3 = random.randint(200,300)

        a_cp = cp.random.random((dim1,rank),dtype=cp.float32)
        b_cp = cp.random.random((dim2,rank),dtype=cp.float32)
        c_cp = cp.random.random((dim3,rank),dtype=cp.float32)

        #a_np = cp.asnumpy(a_cp.copy())
        #b_np = cp.asnumpy(b_cp.copy())
        #c_np = cp.asnumpy(c_cp.copy())

        A1,A2,A3,X = init_tensor(dim1,dim2,dim3,rank)
        #np.save('Benchmark_files/TallSkinnyFactorMatrices_a.npy', a_np)
        #np.save('Benchmark_files/TallSkinnyFactorMatrices_b.npy', b_np)
        #np.save('Benchmark_files/TallSkinnyFactorMatrices_c.npy', c_np)
        ## Time CUDA
        start_time_cuda = time.time()
        S = decompose_QR_KERNELS(X,rank=rank,factor_matrices=[a_cp,b_cp,c_cp],max_iter=max_iterations,verbose=False)
        end_time_cuda = time.time()

        #X = cp.asnumpy(X.copy())

        #start_time_serial = time.time()
        #S = SU.decompose_QR_SERIAL(X,rank=rank,factor_matrices=[a_np,b_np,c_np],max_iter=max_iterations,verbose=False)
        #end_time_serial = time.time()

        print("TEST NUMBER: ",test)
        print("Rank: ",rank)
        print("Dim1: ",dim1)
        print("Dim2: ",dim2)
        print("Dim3: ",dim3)
        print("Cuda Time: ",end_time_cuda-start_time_cuda)
        #print("Serial Time: ",end_time_serial-start_time_serial)

 
"""
print("\nSHORT FAT DIMENSIONS",flush=True)
for test in range(num_tests):

    
    rank = random.randint(500,1000)
    dim1 = random.randint(10,50)
    dim2 = random.randint(10,50)
    dim3 = random.randint(10,50)

    a_cp = cp.random.random((dim1,rank),dtype=cp.float32)
    b_cp = cp.random.random((dim2,rank),dtype=cp.float32)
    c_cp = cp.random.random((dim3,rank),dtype=cp.float32)

    a_np = cp.asnumpy(a_cp.copy())
    b_np = cp.asnumpy(b_cp.copy())
    c_np = cp.asnumpy(c_cp.copy())
    np.save('Benchmark_files/ShortFatFactorMatrices_a.npy',a_np)
    np.save('Benchmark_files/ShortFatFactorMatrices_b.npy',b_np)
    np.save('Benchmark_files/ShortFatFactorMatrices_c.npy',c_np)
    A1,A2,A3,X = init_tensor(dim1,dim2,dim3,rank)

    ## Time CUDA
    start_time_cuda = time.time()
    S = decompose_QR_KERNELS(X,rank=rank,factor_matrices=[a_cp,b_cp,c_cp],max_iter=max_iterations,verbose=False)
    end_time_cuda = time.time()

    X = cp.asnumpy(X.copy())

    start_time_serial = time.time()
    S = SU.decompose_QR_SERIAL(X,rank=rank,max_iter=max_iterations,factor_matrices=[a_np,b_np,c_np],verbose=False)
    end_time_serial = time.time()

    print("TEST NUMBER: ",test)
    print("Rank: ",rank)
    print("Dim1: ",dim1)
    print("Dim2: ",dim2)
    print("Dim3: ",dim3)
    print("Cuda Time: ",end_time_cuda-start_time_cuda)
    print("Serial Time: ",end_time_serial-start_time_serial)

"""