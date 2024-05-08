#include <stdlib.h>
//#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include "utils.cpp"
void init_data(int dim1, int dim2,int dim3, int rank,Tensor X, Matrix A1,Matrix A2,
                  Matrix A3);
void DecomposeQR(Tensor X, Matrix* As, Matrix* Qs, Matrix* Rs);
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
      A1.M = A1_;
      A2.M = A2_;
      A3.M = A3_;
      X.X = X_;
      KR1.M = KR_out1;
      KR2.M = KR_out2;
      KR3.M = KR_out3;

      khatri_rao(A2.M,A3.M,KR1.M,dim2,dim3,rank);
      khatri_rao(A1.M,A3.M,KR2.M,dim1,dim3,rank);
      khatri_rao(A1.M,A2.M,KR3.M,dim1,dim2,rank);

      Matrix test = Tensor_Unfold(X,0);
      KR1.print();
      //matrix_multiply(dim2*dim3,rank,dim2*dim3)
      self_multiply(KR1.M,out,rank,dim2*dim3);
      matrix_multiply(dim1,dim2*dim3,rank, test.M,KR1.M, out_KR);
      OUT_KR.M = out_KR;
     // OUT_KR.print();
      OUT.M = out;
     // OUT.print();
      cholesky(out,L,rank);
      L1.M = L;
      //L1.print();
      OUT_KR.print();
      L1.print();
      forward_substitution(rank,dim1,L,OUT_KR.M,backsolve);
      OUT_KR.M = backsolve;
      OUT_KR.print();

}
void init_data(int dim1, int dim2,int dim3, int rank,Tensor X, Matrix A1,Matrix A2,Matrix A3){
      FILE *A1_file = fopen("C_testfunctions/A1.bin", "rb");
      FILE *A2_file = fopen("C_testfunctions/A2.bin", "rb");
      FILE *A3_file = fopen("C_testfunctions/A3.bin", "rb");
      FILE *X_file = fopen("C_testfunctions/X.bin", "rb");

      // From file
      float* A1_ = (float*)calloc(dim1*rank,sizeof(float));
      float* A2_ = (float*)calloc(dim2*rank,sizeof(float));
      float* A3_ = (float*)calloc(dim3*rank,sizeof(float));
      float* X_ = (float*)calloc(dim1*dim2*dim3,sizeof(float));

      file_to_ptr(A1_file,dim1*rank,A1_);
      file_to_ptr(A2_file,dim2*rank,A2_);
      file_to_ptr(A3_file,dim3*rank,A3_);
      file_to_ptr(X_file,dim1*dim2*dim3,X_);
      A1.M = A1_;
      A2.M = A2_;
      A3.M = A3_;
      X.X = X_;

}


void DecomposeQR(Tensor X, Matrix* As, Matrix* Qs, Matrix* Rs){
      QR(As[0].M,Qs[0].M,Rs[0].M,As[0].row_dim,As[0].column_dim);



}

void initialize_data(int dim1, int dim2,int dim3,int rank){
      return;
}