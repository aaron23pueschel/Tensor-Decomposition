import cupy as cp


def ttm_launcher(A,B,X,out1,out2,dim1,dim2,dim3,rank,kernels,skip=1,num_kernels = 8):
    """
    Returns the ttm operation for a given mode.

    Parameters
    ----------
    A : cupy.ndarray
        Factor matrix A.
    B : cupy.ndarray
        Factor matrix B.
    X : cupy.ndarray
        Tensor X.
    out1 : cupy.ndarray
        Output 1.
    out2 : cupy.ndarray
        Output 2.
    dim1 : int
        Dimension 1.
    dim2 : int
        Dimension 2.
    dim3 : int
        Dimension 3.
    rank : int
        Rank of the tensor.
    kernels : function
        Compiled kernel functions.
    skip : int, optional
        Dimension to skip. Defaults to 1.
    num_kernels : int, optional
        This is actually blocksize.

    Returns
    -------
    None
    """
    min_rank2 = min([rank,dim2])
    min_rank1 = min([rank,dim1])
    min_rank3 = min([rank,dim3])

    
    block_dim1 = dim1//(num_kernels)+1
    block_dim2 = dim2//(num_kernels)+1
    block_dim3 = dim3//(num_kernels)+1

    block_rank1 = min_rank1//(num_kernels)+1
    block_rank2 = min_rank2//(num_kernels)+1
    block_rank3 = min_rank3//(num_kernels)+1
    max_dim = (num_kernels,num_kernels,num_kernels)

    
    

    if skip == 1:
            kernels[1]((block_dim1,block_rank2,block_dim3),max_dim,(A,X,out1,dim1,dim2,dim3,min_rank2))
            kernels[2]((block_dim1,block_rank2,block_rank3),max_dim,(B,out1,out2,dim1,min_rank2,dim3,min_rank3))
   
    if skip == 2:

            kernels[0]((block_rank1,block_dim2,block_dim3),max_dim,(A,X,out1,dim1,dim2,dim3,min_rank1))
            kernels[2]((block_rank1,block_dim2,block_rank3),max_dim,(B,out1,out2,min_rank1,dim2,dim3,min_rank3))

    if skip == 3:
            kernels[0]((block_rank1,block_dim2,block_dim3),max_dim,(A,X,out1,dim1,dim2,dim3,min_rank1))
            kernels[1]((block_rank1,block_rank2,block_dim3),max_dim,(B,out1,out2,min_rank1,dim2,dim3,min_rank2))




def ttm1_kernel(type_ = "naive"):
    """
    Returns the ttm1 CUDA kernel.

    Parameters
    ----------
    type_ : str, optional
        Type of the ttm1 CUDA kernel. Defaults to "naive".

    Returns
    -------
    cupy.RawKernel
        ttm1 function.
    """
    if type_ =="naive":
        ttm_kernel  = cp.RawKernel(r'''

        extern "C" __global__

        void ttm_1(const float* Q1, float* X, float* out, int dim1,int dim2,int dim3,int rank) {
            int w = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;
            int k = threadIdx.z + blockIdx.z * blockDim.z;
            int threadId = threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);     
            if(w<rank && j<dim2 && k<dim3){            
                float temp = 0;               
                for(int i=0;i<dim1;i++)
                    temp+= Q1[i*rank+w]*X[i* dim2*dim3 + (j * dim3) + k]; 
                out[w*dim2*dim3+j*dim3+k] = temp;
            }
        }''', 'ttm_1')



        return ttm_kernel
    

    ## BROKEN DO NOT USE
    elif type_ == "shared":
        ttm_kernel  = cp.RawKernel(r'''

            extern "C" __global__

            void ttm_1(float* Q1, float* X, float* out, int dim1,int dim2,int dim3,int rank) {
                const int BLOCKSIZE = 8;
                
                
                float* Q1_temp = Q1;
                float* X_temp = X;
               
                __shared__ float Xs[BLOCKSIZE][BLOCKSIZE][BLOCKSIZE];                                            
                __shared__ float Qs[BLOCKSIZE*BLOCKSIZE];                            
                int w = threadIdx.x + blockIdx.x * blockDim.x;
                int j = threadIdx.y + blockIdx.y * blockDim.y;
                int k = threadIdx.z + blockIdx.z * blockDim.z;
                float sum = 0.0f;
                Q1 +=  (blockIdx.x)*rank*BLOCKSIZE;
                X += blockIdx.x*dim2*dim3*BLOCKSIZE;
                __syncthreads(); 
                                              
                for(int C=0; C<dim1/BLOCKSIZE+1;C++){
                   
                    __syncthreads();  // Make sure we are starting at the same place

                    int i_shared = threadIdx.x + blockIdx.x * blockDim.x;
                    if(i_shared*rank+w < rank*dim1)
                        Qs[threadIdx.y*BLOCKSIZE+threadIdx.x] = Q1[threadIdx.y*rank+threadIdx.x];
                    else 
                        Qs[threadIdx.y*BLOCKSIZE+threadIdx.x] = 0.0f;    
                    
                                                                                                        
                    if(i_shared<dim1 && j<dim2 && k<dim3)
                        Xs[threadIdx.x][threadIdx.y][threadIdx.z]=X[threadIdx.x* dim2*dim3 + (j * dim3) + k];
                    else
                        Xs[threadIdx.x][threadIdx.y][threadIdx.z] = 0.0f;
        
                        
                    
                    
                    __syncthreads(); //Wait for threads to finish writing
                    X +=  BLOCKSIZE*dim2*dim3;
                    Q1 +=  BLOCKSIZE*rank;                        
                    __syncthreads();
                    if(w<rank && j<dim2 && k<dim3){  
                        float temp = 0.0f;
                        int min_ = (dim1 < BLOCKSIZE) ? dim1 : BLOCKSIZE;                
                        for(int i=0;i<BLOCKSIZE;i++){
                            temp+= Qs[i*BLOCKSIZE+ threadIdx.x]*Xs[i][threadIdx.y][threadIdx.z]; // Q may be transposed. if so, switch i and w in this line
                          //  printf("%f %f\n",Xs[i][threadIdx.y][threadIdx.z],X_temp[i* dim2*dim3 + (j * dim3) + k]);
                          //  printf("%f %f %d\n",Qs[i*BLOCKSIZE+threadIdx.x],Q1_temp[i*rank+w],i);
                            
                        }
                        sum += temp;
                        __syncthreads();
                    }
                     __syncthreads();
                }
                 __syncthreads();
               // if(w<rank && j<dim2 && k<dim3)  {                         
                    out[w*dim2*dim3+j*dim3+k] += sum; 
                __syncthreads();                                                           
               // printf("%f\n",out[w*dim2*dim3+j*dim3+k]);
                
               // }
           __syncthreads();
             
            }''', 'ttm_1')
        return ttm_kernel
       

    



def ttm2_kernel():
    """
    Returns the CUDA ttm2 kernel.

    Returns
    -------
    cupy.RawKernel
        ttm2 CUDA function.
    """

    ttm_kernel  = cp.RawKernel(r'''

    extern "C" __global__

    void ttm_2(const float* Q2, float* X, float* out, int dim1,int dim2,int dim3,int rank) {
                             
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int w = threadIdx.y + blockIdx.y * blockDim.y;
            int k = threadIdx.z + blockIdx.z * blockDim.z;
            if(i<dim1 && w<rank && k<dim3){                      
                float temp = 0;               
                for(int j=0;j<dim2;j++){
                    temp+= Q2[j*rank+w]*X[i* dim2*dim3 + (j * dim3) + k]; // Q may be transposed. if so, switch i and w in this line
                }
            
                out[i*rank*dim3+w*dim3+k] = temp;

            }
    }''', 'ttm_2')
    return ttm_kernel



def ttm3_kernel():
    """
    Returns the ttm3 CUDA kernel.

    Returns
    -------
    cupy.RawKernel
        ttm3 kernel.
    """
    ttm_kernel  = cp.RawKernel(r'''

    extern "C" __global__

    void ttm_3(const float* Q3, float* X, float* out, int dim1,int dim2,int dim3,int rank) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        int w = threadIdx.z + blockIdx.z * blockDim.z;
        if(i<dim1 && j<dim2 && w<rank){                      
            float temp = 0;               
            for(int k=0;k<dim3;k++)
                temp+= Q3[k*rank+w]*X[i* dim2*dim3 + (j * dim3) + k]; // Q may be transposed. if so, switch i and w in this line
            out[i*dim2*rank+j*rank+w] = temp;
        }
    }''', 'ttm_3')
    return ttm_kernel


def khatriRao_kernel():
    """
    Returns the CUDA Khatri Rao kernel.

    Returns
    -------
    cupy.RawKernel
        Khatri Rao kernel.
    """
    khatriRao  = cp.RawKernel(r'''

    extern "C" __global__

    void khatriRao(const float* M1, const float* M2, float* out, int dim1,int dim2,int rank) {
        int i = threadIdx.y + blockIdx.y * blockDim.y;
        int r = threadIdx.x + blockIdx.x * blockDim.x;
        //printf("%d %d\n",blockDim.y,blockIdx.y);
        if(r<rank && i<dim1){                    
            for(int j=0;j<dim2;j++){
                out[i*dim2*rank+j*rank+r]=M1[i+dim1*r]*M2[j+dim2*r];  
                
            }
        }




    }''', 'khatriRao')
    return khatriRao

