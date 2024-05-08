
#include <cuda.h>
#include "cudaHeader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>



__global__ void MatSelfKernel(float* A,float* out,int dim1,int dim2) {

    int i = blockIdx.y*blockDim.y+threadIdx.y;
    int j = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;
    printf("dim1: %d dim2: %d\n",i,j);
    if (i < dim1 && j < dim1) {
        // each thread computes one element of the block sub-matrix
        for (int k = 0; k < dim2; k++) {
            tmpSum += A[k*dim1+i]*A[k*dim1+j];
        }
    }
    out[i * dim1 + j] = tmpSum;
    printf("%f",tmpSum);
}


void MatSelfLauncher(float* A,float* out,int dim1,int dim2){
    const int blockSize = 256;
    const int numBlocks = (dim1 + blockSize - 1) / blockSize;
    MatSelfKernel<<<numBlocks,blockSize>>>(A,out,dim1,dim2);
    cudaError_t kernelLaunchError = cudaGetLastError();
    if (kernelLaunchError != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(kernelLaunchError));
        return;
    }
    cudaDeviceSynchronize();
}