#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void khatri_rao(float* M1,float* M2,float* out_M, int mdim1,int mdim2,int rank){
    for(int r=0;r<rank;r++){
        for(int i =0; i<mdim1;i++){
            for(int j=0;j<mdim2;j++){
                out_M[i*mdim2*rank+j*rank+r]=M1[r+i*rank]*M2[r+j*rank];  
                
            }
        }       
    }
}

float vector_dot(float *a, float *b, int L) {
    float mag = 0.0f;
    for (int i = 0; i < L; i++)
        mag += a[i] * b[i];

    return mag;
}

float vector_mag(float *vector, int L) {
    float dot = vector_dot(vector, vector, L);
    return sqrtf(dot);
}

float *vector_proj(float *a, float *b, float *out, int L) {
    float num = vector_dot(a, b, L);
    float deno = vector_dot(b, b, L);
    for (int i = 0; i < L; i++)
        out[i] = num * b[i] / deno;

    return out;
}

float *vector_sub(float *a, float *b, float *out, int L) {
    for (int i = 0; i < L; i++)
        out[i] = a[i] - b[i];

    return out;
}
 
void QR(float *A, float *Q, float *R,int M,int N) {
    float *tmp_vector = (float*) malloc(M * sizeof(float));
    float *col_vector = (float*) malloc(M * sizeof(float));
    float *col_vector2 = (float*) malloc(M * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            R[i * N + j] = 0.0f;
        }

        for (int j = 0; j < M; j++) {
            tmp_vector[j] = A[j * N + i];
            col_vector[j] = A[j * N + i];
        }

        for (int j = 0; j < i; j++) {
            for (int k = 0; k < M; k++) {
                col_vector2[k] = Q[k * N + j];
            }
            vector_proj(col_vector, col_vector2, col_vector2, M);
            vector_sub(tmp_vector, col_vector2, tmp_vector, M);
        }

        float mag = vector_mag(tmp_vector, M);
        for (int j = 0; j < M; j++) {
            Q[j * N + i] = tmp_vector[j] / mag;
        }

        for (int kk = 0; kk < M; kk++) {
            col_vector[kk] = Q[kk * N + i];
        }

        for (int k = i; k < N; k++) {
            for (int kk = 0; kk < M; kk++) {
                col_vector2[kk] = A[kk * N + k];
            }
            R[i * N + k] = vector_dot(col_vector, col_vector2, M);
        }
    }


}

void TTM_mode1(float* M, float* X, float* out,int dim1,int dim2, int dim3, int rank){
    
    for(int w=0;w<rank;w++){
        for(int j=0;j<dim2;j++){
            for(int k=0;k<dim3;k++){
                float temp = 0.0f;
                for(int i=0;i<dim1;i++){
                    temp+= M[i*rank +w]*X[i* dim2*dim3 + (j * dim3) + k];
                } 
                out[w*dim2*dim3+j*dim3+k] = temp;     
            }
        }
    }
}
void TTM_mode3(float* M, float* X, float* out,int dim1,int dim2, int dim3, int rank){
 
    for(int i=0;i<dim1;i++){
        for(int j=0;j<dim2;j++){
            for(int w=0;w<rank;w++){
                float temp = 0.0f;
                for(int k=0;k<dim3;k++){
                    temp+= M[k*rank+w]*X[i* dim2*dim3 + (j * dim3) + k];
                }
                out[i*dim2*rank+j*rank+w] = temp;
            }
        }
    }
}
void TTM_mode2(float* M, float* X, float* out,int dim1,int dim2, int dim3, int rank){

    for(int i=0;i<dim1;i++){
        for(int w=0;w<rank;w++){
            for(int k=0;k<dim3;k++){
                float temp = 0.0f;
                for(int j=0;j<dim2;j++){
                    temp+= M[j*rank+w]*X[i* dim2*dim3 + (j * dim3) + k];
                    
                }
                
                out[i*rank*dim3+w*dim3+k] = temp;
            }
        }
    }
}


