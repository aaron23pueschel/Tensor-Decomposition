import scipy
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from np_utils import *

class Serial_C:
    """
    Class to initialize all the functions in utils.c
    """
    def __init__(self):
        self.functions = {}
        try:
            self.funcs = ctypes.cdll.LoadLibrary("./naive_c.so")
        except:
            raise FileNotFoundError("Compile utils.c: See documentation")
        self.float_pointer_3d = ndpointer(dtype=np.float32,ndim=3, flags="C_CONTIGUOUS")
        self.float_pointer_2d = ndpointer(dtype=np.float32,ndim=2, flags="C_CONTIGUOUS")
    def khatri_rao(self):
        
        KR = self.funcs.khatri_rao
        KR.restype = None
       
        KR.argtypes = [self.float_pointer_2d,self.float_pointer_2d,self.float_pointer_2d,ctypes.c_int,ctypes.c_int,ctypes.c_int]
        return KR
    def TTM_mode1(self):
        KR = self.funcs.TTM_mode1
        KR.restype = None
       
        KR.argtypes = [self.float_pointer_2d,self.float_pointer_3d,self.float_pointer_3d,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
        return KR
    def TTM_mode2(self):
        KR = self.funcs.TTM_mode2
        KR.restype = None
        
        KR.argtypes = [self.float_pointer_2d,self.float_pointer_3d,self.float_pointer_3d,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
        return KR
    def TTM_mode3(self):
        KR = self.funcs.TTM_mode3
        KR.restype = None
    
        KR.argtypes = [self.float_pointer_2d,self.float_pointer_3d,self.float_pointer_3d,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
        return KR
    def QR(self):
        KR = self.funcs.QR
        KR.restype = None
        KR.argtypes = [self.float_pointer_2d,self.float_pointer_2d,self.float_pointer_2d,ctypes.c_int,ctypes.c_int]
        return KR

def init_tensor(dim1,dim2,dim3,rank,noise=0):
    
    A1 = np.random.random((dim1,rank),dtype=np.float32)
    A2 = np.random.random((dim2,rank),dtype=np.float32)
    A3 = np.random.random((dim3,rank),dtype=np.float32)

    sum = np.zeros((dim1,dim2,dim3))
    for i in range(rank):
        sum+= np.einsum('i,j,k->ijk',A1[:,i],A2[:,i],A3[:,i])
    X=sum
    if noise!=0:
        X  += np.random.normal(0,noise,size=(dim1,dim2,dim3)) 
    X = X.astype(np.float32)

    return A1,A2,A3,X
def multi_ttm(A,B,X,out1,out2,dim1,dim2,dim3,rank,kernels,skip=1):
    min_rank2 = min([rank,dim2])
    min_rank1 = min([rank,dim1])
    min_rank3 = min([rank,dim3])
    if skip == 1:
        kernels[1](A,X,out1,dim1,dim2,dim3,min_rank2)
        kernels[2](B,out1,out2,dim1,min_rank2,dim3,min_rank3)
    if skip == 2:
        kernels[0](A,X,out1,dim1,dim2,dim3,min_rank1)
        kernels[2](B,out1,out2,min_rank1,dim2,dim3,min_rank3)
    if skip == 3:
        kernels[0](A,X,out1,dim1,dim2,dim3,min_rank1)
        kernels[1](B,out1,out2,min_rank1,dim2,dim3,min_rank2)

def QR_serial(A,Q,R,QR_func,dtype=np.float32):
    m = A.shape[0]
    n = A.shape[1]
    if m >= n:
        QR_func(A,Q,R,m,n)
        return
    R_hat = np.zeros((m,m),dtype=np.float32)
    QR_func(A[0:m,0:m].copy(),Q,R_hat,m,m)

    R[0:m,0:m] = R_hat
    R[:,m:] = Q.T@A[:,m:]


def init_tensor(dim1,dim2,dim3,rank,noise=0):
    A1 = np.random.random((dim1,rank)).astype(np.float32)
    A2 = np.random.random((dim2,rank)).astype(np.float32)
    A3 = np.random.random((dim3,rank)).astype(np.float32)

    sum = np.zeros((dim1,dim2,dim3))
    for i in range(rank):
        sum+= np.einsum('i,j,k->ijk',A1[:,i],A2[:,i],A3[:,i])
    X=sum
    if noise!=0:
        X  += np.random.normal(0,noise,size=(dim1,dim2,dim3)) 
    X = X.astype(np.float32)

    return A1,A2,A3,X

def decompose_QR_SERIAL(X, factor_matrices = None,rank=10, max_iter=101, rel_tol = 10e-6,verbose=True):
    if factor_matrices is None:
        a = np.random.random((X.shape[0],rank))
        b = np.random.random((X.shape[1],rank))
        c = np.random.random((X.shape[2],rank))
    else:
        a = factor_matrices[0]
        b = factor_matrices[1]
        c = factor_matrices[2]
    
    functions = Serial_C()
    ttm_funcs = [functions.TTM_mode1(),functions.TTM_mode2(),functions.TTM_mode3()]
    khatri_rao_serial = functions.khatri_rao()
    QR_c = functions.QR()
    dim1 = X.shape[0]
    dim2 = X.shape[1]
    dim3 = X.shape[2]

    min_rank1 = min([rank,dim1])
    min_rank2 = min([rank,dim2])
    min_rank3 = min([rank,dim3])
    
    ttm1_first = np.zeros((dim1,min_rank2,dim3),dtype=np.float32)
    ttm1_second = np.zeros((dim1,min_rank2,min_rank3),dtype=np.float32)

    ttm2_first = np.zeros((min_rank1,dim2,dim3),dtype=np.float32)
    ttm2_second = np.zeros((min_rank1,dim2,min_rank3),dtype=np.float32)

    ttm3_first = np.zeros((min_rank1,dim2,dim3),dtype=np.float32)
    ttm3_second = np.zeros((min_rank1,min_rank2,dim3),dtype=np.float32)
    
    a = np.random.random((dim1,rank)).astype(np.float32)
    b = np.random.random((dim2,rank)).astype(np.float32)
    c = np.random.random((dim3,rank)).astype(np.float32)

    Qa,Ra = np.linalg.qr(a)
    Qb,Rb = np.linalg.qr(b)
    Qc,Rc = np.linalg.qr(c)
    

    KR_ab = np.zeros((min_rank2*min_rank1,rank),dtype=np.float32) ## V0
    KR_ac = np.zeros((min_rank1*min_rank3,rank),dtype=np.float32) ## V1
    KR_bc = np.zeros((min_rank3*min_rank2,rank),dtype=np.float32) ## V2
    
    
    Q0 = np.zeros((min_rank2*min_rank3,rank),dtype=np.float32)
    R0 = np.zeros((rank,rank),dtype=np.float32)

    Q1 = np.zeros((min_rank1*min_rank3,rank),dtype=np.float32)
    R1 = np.zeros((rank,rank),dtype=np.float32)

    Q2 = np.zeros((min_rank2*min_rank1,rank),dtype=np.float32)
    R2 = np.zeros((rank,rank),dtype=np.float32)


    res_a1 = np.linalg.norm(khatri_rao(c, b)@(a.T) - f_unfold(X,mode=0).T,'fro')
    res_b1 = np.linalg.norm(khatri_rao(c, a)@(b.T) - f_unfold(X,mode=1).T,'fro')
    res_c1 = np.linalg.norm(khatri_rao(b, a)@(c.T) - f_unfold(X,mode=2).T,'fro')

    
    Ra = Ra.astype(np.float32)
    Rb = Rb.astype(np.float32)
    Rc = Rc.astype(np.float32)

    for epoch in range(max_iter):
        ## FIRST ITERATION
        khatri_rao_serial(Rc,Rb,KR_bc,min_rank3,min_rank2,rank)
        Q0,R0 = np.linalg.qr(KR_bc)
        #QR_serial(KR_bc,Q0,R0,QR_func=QR_c)
        multi_ttm(Qb,Qc,X,ttm1_first,ttm1_second,dim1,dim2,dim3,rank,ttm_funcs,skip=1)
        Y0 = f_unfold(ttm1_second,mode=0)
        W = Y0@Q0
        if R0.shape[0]!=R0.shape[1]:
            a = (linalg.solve_triangular(R0[:,0:W.T.shape[0]],W.T).T)@R0
        else:
            a = linalg.solve_triangular(R0,W.T).T
        Qa,Ra = np.linalg.qr(a)
        #QR_serial(a,Qa,Ra,QR_c)


        ##SECOND ITERATION
        khatri_rao_serial(Rc,Ra,KR_ac,min_rank3,min_rank1,rank)
        Q1,R1 = np.linalg.qr(KR_ac)
        #QR_serial(KR_ac,Q1,R1,QR_func=QR_c)
        multi_ttm(Qa,Qc,X,ttm2_first,ttm2_second,dim1,dim2,dim3,rank,ttm_funcs,skip=2)
        #ttm2_second = ttm_rank2(Qa,Qc,X,skip=2)
        Y0 = f_unfold(ttm2_second,mode=1)
        W= Y0@Q1
        if R1.shape[0]!=R1.shape[1]:
            b = (linalg.solve_triangular(R1[:,0:W.T.shape[0]],W.T).T)@R1
        else:
            b = linalg.solve_triangular(R1,W.T).T
        Qb,Rb = np.linalg.qr(b)
        #QR_serial(b,Qb,Rb,QR_c)

        
        ## THIRD ITERATION
        khatri_rao_serial(Rb,Ra,KR_ab,min_rank2,min_rank1,rank)
        
        Q2,R2 = np.linalg.qr(KR_ab)
        #QR_serial(KR_ab,Q2,R2,QR_func=QR_c)
        multi_ttm(Qa,Qb,X,ttm3_first,ttm3_second,dim1,dim2,dim3,rank,ttm_funcs,skip=3)
        #ttm3_second = ttm_rank2(Qa,Qb,X,skip=3)
        Y0 = f_unfold(ttm3_second,mode=2)
        W= Y0@Q2
        if R2.shape[0]!=R2.shape[1]:
            c = (linalg.solve_triangular(R2[:,0:W.T.shape[0]],W.T).T)@R2
        else:
            c = linalg.solve_triangular(R2,W.T).T
        Qc,Rc = np.linalg.qr(c)
        #QR_serial(c,Qc,Rc,QR_c)

        

        
        #print(f"Epoch# {epoch} complete")
        if verbose:
            res_a = np.linalg.norm(khatri_rao(c, b)@(a.T) - f_unfold(X,mode=0).T,'fro')
            res_b = np.linalg.norm(khatri_rao(c, a)@(b.T) - f_unfold(X,mode=1).T,'fro')
            res_c = np.linalg.norm(khatri_rao(b, a)@(c.T) - f_unfold(X,mode=2).T,'fro')
            print("Epoch:", epoch, "MAX Rel Error: ", max([res_a/res_a1,res_b/res_b1,res_c/res_c1]))
            #print("Epoch:", epoch, "| Loss (A):", res_a, "| Loss (B):", res_b, "| Loss (C):", res_c)
            if max([res_a/res_a1,res_b/res_b1,res_c/res_c1])< rel_tol:
                print(f"ALS converged in {epoch} iterations")
                break
            
    #print("Epoch:", epoch, "Rel Error: ", max([res_a/res_a1,res_b/res_b1,res_c/res_c1]))
    return a.T, b.T, c.T