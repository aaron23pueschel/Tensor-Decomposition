#include <stdio.h>
#include <math.h>
#include "utils.h"
#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include <stdlib.h>
#include "cudaHeader.h"

int main(){
      int dim1 = 5;
      int dim2 = 3;
      int dim3 = 4;
      int rank = 2;
      Tensor X(dim1,dim2,dim3);
      Matrix A1(rank,dim1);
      Matrix A2(rank,dim2);
      Matrix A3(rank,dim3);
      Matrix KR1(dim2*dim3,rank);
      Matrix KR2(dim1*dim3,rank);
      Matrix KR3(dim2*dim1,rank);
      Matrix OUT(rank,rank);
      Matrix OUT_KR(dim1,rank);
      Matrix L1(rank,rank);

      FILE *A1_file = fopen("C_testfunctions/A1.bin", "rb");
      FILE *A2_file = fopen("C_testfunctions/A2.bin", "rb");
      FILE *A3_file = fopen("C_testfunctions/A3.bin", "rb");
      FILE *X_file = fopen("C_testfunctions/X.bin", "rb");

      // From file
      float* A1_ = (float*)calloc(dim1*rank,sizeof(float));
      float* A2_ = (float*)calloc(dim2*rank,sizeof(float));
      float* A3_ = (float*)calloc(dim3*rank,sizeof(float));
      float* X_ = (float*)calloc(dim1*dim2*dim3,sizeof(float));
      float* KR_out1 = (float*)calloc(dim2*dim3*rank,sizeof(float));
      float* KR_out2 = (float*)calloc(dim1*dim3*rank,sizeof(float));
      float* KR_out3 = (float*)calloc(dim2*dim1*rank,sizeof(float));
      float* out = (float*)calloc(rank*rank,sizeof(float));
      float* backsolve = (float*)calloc(dim1*rank,sizeof(float));
      float* out_KR = (float*)calloc(dim1*rank,sizeof(float));
      float* L = (float*)calloc(rank*rank,sizeof(float));
      file_to_ptr(A1_file,dim1*rank,A1_);
      file_to_ptr(A2_file,dim2*rank,A2_);
      file_to_ptr(A3_file,dim3*rank,A3_);
      file_to_ptr(X_file,dim1*dim2*dim3,X_);


     // Cuda pointers 
     float* A1_d, *A2_d, *A3_d, *X_d, *KR_out1_d, *KR_out2_d, *KR_out3_d, *out_d, *backsolve_d, *out_KR_d, *L_d;

    cudaMalloc((void**)&A1_d, dim1 * rank * sizeof(float));
    cudaMalloc((void**)&A2_d, dim2 * rank * sizeof(float));
    cudaMalloc((void**)&A3_d, dim3 * rank * sizeof(float));
    cudaMalloc((void**)&X_d, dim1 * dim2 * dim3 * sizeof(float));
    cudaMalloc((void**)&KR_out1_d, dim2 * dim3 * rank * sizeof(float));
    cudaMalloc((void**)&KR_out2_d, dim1 * dim3 * rank * sizeof(float));
    cudaMalloc((void**)&KR_out3_d, dim2 * dim1 * rank * sizeof(float));
    cudaMalloc((void**)&out_d, rank * rank * sizeof(float));
    cudaMalloc((void**)&backsolve_d, dim1 * rank * sizeof(float));
    cudaMalloc((void**)&out_KR_d, dim1 * rank * sizeof(float));
    cudaMalloc((void**)&L_d, rank * rank * sizeof(float));

    cudaMemcpy(A1_d, A1_, dim1 * rank * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(A2_d, A2_, dim2 * rank * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(A3_d, A3_, dim3 * rank * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(X_d, X_, dim1 * dim2 * dim3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(KR_out1_d, KR_out1, dim2 * dim3 * rank * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(KR_out2_d, KR_out2, dim1 * dim3 * rank * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(KR_out3_d, KR_out3, dim2 * dim1 * rank * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, rank * rank * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(backsolve_d, backsolve, dim1 * rank * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_KR_d, out_KR, dim1 * rank * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(L_d, L, rank * rank * sizeof(float), cudaMemcpyHostToDevice);



    A1.M = A1_d;
    A2.M = A2_d;
    A3.M = A3_d;
    X.X = X_d;
    KR1.M = KR_out1_d;
    KR2.M = KR_out2_d;
    KR3.M = KR_out3_d;


    MatSelfLauncher(A1_d,out,rank,dim1);
    cudaMemcpy(L,out, rank * rank * sizeof(float), cudaMemcpyDeviceToHost);
    L1.M = out;
    L1.print();
    


}
