
import cupy as cp
from cupyx.scipy import linalg
from cp_kernels import *

def init_tensor(dim1,dim2,dim3,rank,noise=0):
    """
    Initializes tensor of order 3.

    Parameters
    ----------
    dim1 : int
        Size of the first dimension.
    dim2 : int
        Size of the second dimension.
    dim3 : int
        Size of the third dimension.
    rank : int
        Rank of the tensor.
    noise : int, optional
        Added noise constant. Defaults to 0.

    Returns
    -------
    cupy.ndarray
        Factor matrices, Tensor.
    """

    A1 = cp.random.random((dim1,rank),dtype=cp.float32)
    A2 = cp.random.random((dim2,rank),dtype=cp.float32)
    A3 = cp.random.random((dim3,rank),dtype=cp.float32)

    sum = cp.zeros((dim1,dim2,dim3))
    for i in range(rank):
        sum+= cp.einsum('i,j,k->ijk',A1[:,i],A2[:,i],A3[:,i])
    X=sum
    if noise!=0:
        X  += cp.random.normal(0,noise,size=(dim1,dim2,dim3)) 
    X = X.astype(cp.float32)

    return A1,A2,A3,X

def f_unfold(tensor, mode=0,order='F'):
    """
    Returns unfolded tensor along a particular mode.

    Parameters
    ----------
    tensor : cupy.ndarray
        3D tensor.
    mode : int, optional
        Mode along which to unfold. Defaults to 0.
    order : str, optional
        Ordering of the unfold operation. Defaults to Fortran column major.

    Returns
    -------
    cupy.ndarray
        Unfolded matrix.
    """
    return cp.reshape(cp.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order=order)

def khatri_rao(a,b):
    """
    Returns Khatri Rao product of two factor matrices.

    Parameters
    ----------
    a : cupy.ndarray
        First matrix.
    b : cupy.ndarray
        Second matrix.

    Returns
    -------
    numpy.array
        Khatri Rao product.
    """
    return (a[:,None] * b).reshape(-1,a.shape[1])

def matrixitazion_mode2(X,a,b,mode=1):
    """ Returns the unfolded tensor after tensor matrix product

    DEPRECATED
    """
    if mode==1:
        return cp.einsum('ijk,jw,ky->iwy',X,a,b).reshape(X.shape[mode-1],a.shape[1]*b.shape[1],order='F')
    if mode==2:
        return f_unfold(cp.einsum('ijk,iw,ky->wjy',X,a,b),mode=mode-1)#.reshape(X.shape[mode-1],a.shape[1]*b.shape[1],order='F')
    if mode==3:
        return f_unfold(cp.einsum('ijk,iw,jy->wyk',X,a,b),mode=mode-1)



def ttm_rank2(Q1,Q2,X,skip=1,out=None):
    """
    Returns the tensor matrix product for 2 factor matrices.

    Parameters
    ----------
    Q1 : cp.ndarray
        First factor matrix.
    Q2 : cp.ndarray
        Second factor matrix.
    X : cp.ndarray
        3D tensor.
    skip : int, optional
        Mode along which to perform ttm. Defaults to 1.

    Returns
    -------
    cp.ndarray
        TTM along a particular mode.
    """
     
    if skip == 1:
        return cp.einsum('cj,kl,ijl->ick',Q1, Q2, X) 
    elif skip ==2:
        return cp.einsum('ci,kl,ijl->cjk',Q1, Q2, X)
    elif skip==3:
        return  cp.einsum('ci,jl,ilk->cjk',Q1, Q2,X)


def decompose_normal_eqns(tensor, factor_matrices = None,rank=10, max_iter=101, rel_tol = 10e-10,verbose=True):
    """
    Decomposes the tensor into the appropriate factor matrices using vanilla CPD.

    Parameters
    ----------
    tensor : cupy.ndarray
        The tensor to operate on.
    factor_matrices : list, optional
        (Optional) Factor matrices used in the operation. Defaults to None.
    rank : int, optional
        (Optional) The rank of the tensor decomposition. Defaults to 10.
    max_iter : int, optional
        (Optional) The maximum number of iterations for the operation. Defaults to 101.
    rel_tol : float, optional
        (Optional) The relative tolerance for convergence. Defaults to 1e-10.
    verbose : bool, optional
        (Optional) Whether to print verbose output. Defaults to True.

    Returns
    -------
    cupy.ndarray
        The decomposed factor matrices.
    """
    
    dim1 = tensor.shape[0]
    dim2 = tensor.shape[1]
    dim3 = tensor.shape[2]

    if factor_matrices is None:
        a = cp.random.random((tensor.shape[0],rank))
        b = cp.random.random((tensor.shape[1],rank))
        c = cp.random.random((tensor.shape[2],rank))
    else:
        a = factor_matrices[0]
        b = factor_matrices[1]
        c = factor_matrices[2]

    KR_ab = cp.zeros((dim1*dim2,rank))
    KR_ac = cp.zeros((dim1*dim3,rank))
    KR_bc = cp.zeros((dim2*dim3,rank))





    res_a1 = cp.linalg.norm(khatri_rao(c, b)@(a.T) - f_unfold(tensor,mode=0).T,'fro')
    res_b1 = cp.linalg.norm(khatri_rao(c, a)@(b.T) - f_unfold(tensor,mode=1).T,'fro')
    res_c1 = cp.linalg.norm(khatri_rao(b, a)@(c.T) - f_unfold(tensor,mode=2).T,'fro')
    for epoch in range(max_iter):
        # optimize a
        icput_a = khatri_rao(b,c)
        target_a = f_unfold(tensor, mode=0,order='C').T
        a = cp.linalg.solve(icput_a.T@icput_a, icput_a.T@target_a).T

        # optimize b
        icput_b = khatri_rao(a,c)
        target_b = f_unfold(tensor, mode=1,order='C').T
        b = cp.linalg.solve(icput_b.T@icput_b, icput_b.T@target_b).T

        # optimize c
        icput_c = khatri_rao(a,b)
        target_c = f_unfold(tensor, mode=2,order='C').T
        c = cp.linalg.solve(icput_c.T@icput_c, icput_c.T@target_c).T

        res_a = cp.linalg.norm(icput_a.dot(a.T) - target_a,'fro')
        res_b = cp.linalg.norm(icput_b.dot(b.T) - target_b,'fro')
        res_c = cp.linalg.norm(icput_c.dot(c.T) - target_c,'fro')

        if verbose:
            print("Epoch:", epoch, "Rel Error: ", max([res_a/res_a1,res_b/res_b1,res_c/res_c1]))
            #print("Epoch:", epoch, "| Loss (A):", res_a, "| Loss (B):", res_b, "| Loss (C):", res_c)
        if max([res_a/res_a1,res_b/res_b1,res_c/res_c1])< rel_tol:
            print(f"ALS converged in {epoch} iterations")
            break
    print("Epoch:", epoch, "Rel Error: ", max([res_a/res_a1,res_b/res_b1,res_c/res_c1]))
    return a.T, b.T, c.T



def decompose_QR_NAIVE(X, factor_matrices = None,rank=10, max_iter=101, rel_tol = 10e-10,verbose=True):
    """
    Returns the "naive" ALS-QR factorization for a given tensor.

    Parameters
    ----------
    X : cupy.ndarray
        The input matrix to decompose.
    factor_matrices : cupy.ndarray, optional
        (Optional) Initial factor matrices for the decomposition. Defaults to None.
    rank : int, optional
        (Optional) The rank of the decomposition. Defaults to 10.
    max_iter : int, optional
        (Optional) The maximum number of iterations for the decomposition. Defaults to 101.
    rel_tol : float, optional
        (Optional) The relative tolerance for convergence. Defaults to 1e-10.
    verbose : bool, optional
        (Optional) Whether to print verbose output. Defaults to True.

    Returns
    -------
    cupy.ndarray, cupy.ndarray, cupy.ndarray
        The decomposed factor matrices.
    """
    a = cp.random.random((X.shape[0],rank),dtype=cp.float32)
    b = cp.random.random((X.shape[1],rank),dtype=cp.float32)
    c = cp.random.random((X.shape[2],rank),dtype=cp.float32)

    if factor_matrices==None:
        s=1
    else:
        a = factor_matrices[0]
        b = factor_matrices[1]
        c = factor_matrices[2]
    res_a1 = cp.linalg.norm(khatri_rao(c, b)@(a.T) - f_unfold(X,mode=0).T,'fro')
    res_b1 = cp.linalg.norm(khatri_rao(c, a)@(b.T) - f_unfold(X,mode=1).T,'fro')
    res_c1 = cp.linalg.norm(khatri_rao(b, a)@(c.T) - f_unfold(X,mode=2).T,'fro')
    Qa,Ra = cp.linalg.qr(a)
    Qb,Rb = cp.linalg.qr(b)
    Qc,Rc = cp.linalg.qr(c)
    for epoch in range(max_iter):
        
        V0 = khatri_rao(Rc, Rb)
        Q0,R0 = cp.linalg.qr(V0)
        Y = ttm_rank2(Qb.T,Qc.T,X)
        Y0 = f_unfold(Y,mode=0)
        
        W= Y0@Q0
        a = linalg.solve_triangular(R0,W.T).T
        Qa,Ra = cp.linalg.qr(a)


        V1 = khatri_rao(Rc, Ra)
        Q1,R1 = cp.linalg.qr(V1)
        Y = ttm_rank2(Qa.T,Qc.T,X,skip=2)
        Y0 = f_unfold(Y,mode=1)
        W= Y0@Q1
        b = linalg.solve_triangular(R1,W.T).T
        Qb,Rb = cp.linalg.qr(b)

        V2 = khatri_rao(Rb, Ra)
        Q2,R2 = cp.linalg.qr(V2)
        Y = ttm_rank2(Qa.T,Qb.T,X,skip=3)
        Y0 = f_unfold(Y,mode=2)
        W= Y0@Q2
        c = linalg.solve_triangular(R2,W.T).T
        Qc,Rc = cp.linalg.qr(c)


        

        if verbose:
            res_a = cp.linalg.norm(khatri_rao(c, b)@(a.T) - f_unfold(X,mode=0).T,'fro')
            res_b = cp.linalg.norm(khatri_rao(c, a)@(b.T) - f_unfold(X,mode=1).T,'fro')
            res_c = cp.linalg.norm(khatri_rao(b, a)@(c.T) - f_unfold(X,mode=2).T,'fro')
            print("Epoch:", epoch, "Rel Error: ", max([res_a/res_a1,res_b/res_b1,res_c/res_c1]))
            if max([res_a/res_a1,res_b/res_b1,res_c/res_c1])< rel_tol:
                print(f"ALS converged in {epoch} iterations")
                break

    return a.T, b.T, c.T

def reconstruct_tensor(A1,A2,A3,rank):
    """
    Returns the reconstructed tensor from factor matrices A1, A2, A3.

    Parameters
    ----------
    A1 : cupy.ndarray
        Factor matrix A1.
    A2 : cupy.ndarray
        Factor matrix A2.
    A3 : cupy.ndarray
        Factor matrix A3.
    rank : int
        Rank of the tensor.

    Returns
    -------
    cupy.ndarray
        Reconstructed tensor.
    """
    sum = 0
    for i in range(rank):
        sum+= cp.einsum('i,j,k->ijk',A1[i,:],A2[i,:],A3[i,:])
    return sum


def decompose_QR_KERNELS(X, factor_matrices = None,rank=10, max_iter=101, rel_tol = 10e-6,verbose=True,cutensor_ttm = False):
    """
    Performs CPD-ALS with QR using CUDA kernels.

    Parameters
    ----------
    X : cupy.ndarray
        The tensor to decompose.
    factor_matrices : cupy.ndarray, optional
        (Optional) Initial factor matrices for the decomposition. Defaults to None.
    rank : int, optional
        (Optional) The rank of the tensor. Defaults to 10.
    max_iter : int, optional
        (Optional) The maximum number of iterations for the decomposition. Defaults to 101.
    rel_tol : float, optional
        (Optional) The relative tolerance for convergence. Defaults to 1e-6.
    verbose : bool, optional
        (Optional) Whether to print verbose output. Defaults to True.
    cutensor_ttm : bool, optional
        (Optional) Whether to use cuTensor for tensor times matrix computation. Defaults to False.

    Returns
    -------
    cupy.ndarray
        The decomposed factor matrices, residual history.
    """
    KR = khatriRao_kernel()
    ttm1 = ttm1_kernel()
    ttm2 = ttm2_kernel()
    ttm3 = ttm3_kernel()
    ttm_kernels = [ttm1,ttm2,ttm3]
    dim1 = X.shape[0]
    dim2 = X.shape[1]
    dim3 = X.shape[2]
    ttm1_flag = True  # optimize by doing the largest products first
    ttm2_flag = True
    ttm3_flag = True
    min_rank1 = min([rank,dim1])
    min_rank2 = min([rank,dim2])
    min_rank3 = min([rank,dim3])
    

    ttm1_first = cp.zeros((dim1,min_rank2,dim3),dtype=cp.float32)
    ttm1_second = cp.zeros((dim1,min_rank2,min_rank3),dtype=cp.float32)


    ttm2_first = cp.zeros((min_rank1,dim2,dim3),dtype=cp.float32)
    ttm2_second = cp.zeros((min_rank1,dim2,min_rank3),dtype=cp.float32)


    ttm3_first = cp.zeros((min_rank1,dim2,dim3),dtype=cp.float32)
    ttm3_second = cp.zeros((min_rank1,min_rank2,dim3),dtype=cp.float32)



    if factor_matrices is not None:
        a = factor_matrices[0]
        b = factor_matrices[1]
        c = factor_matrices[2]
    else:
        a = cp.random.random((dim1,rank),dtype=cp.float32)
        b = cp.random.random((dim2,rank),dtype=cp.float32)
        c = cp.random.random((dim3,rank),dtype=cp.float32)

    KR_ab = cp.zeros((min_rank2*min_rank1,rank),dtype=cp.float32) ## V0
    KR_ac = cp.zeros((min_rank1*min_rank3,rank),dtype=cp.float32) ## V1
    KR_bc = cp.zeros((min_rank3*min_rank2,rank),dtype=cp.float32) ## V2
    

    # Initial residuals
    res_a1 = cp.linalg.norm(khatri_rao(c, b)@(a.T) - f_unfold(X,mode=0).T,'fro')
    res_b1 = cp.linalg.norm(khatri_rao(c, a)@(b.T) - f_unfold(X,mode=1).T,'fro')
    res_c1 = cp.linalg.norm(khatri_rao(b, a)@(c.T) - f_unfold(X,mode=2).T,'fro')
    Qa,Ra = cp.linalg.qr(a)
    Qb,Rb = cp.linalg.qr(b)
    Qc,Rc = cp.linalg.qr(c)

    Ra = Ra.astype(cp.float32)
    Rb = Rb.astype(cp.float32)
    Rc = Rc.astype(cp.float32)

    Qa = cp.ascontiguousarray(Qa)
    Qb = cp.ascontiguousarray(Qb)
    Qc = cp.ascontiguousarray(Qc)
    KR_blocks = (rank//32+1,rank//32+1,)
    KR_blockdim = (32,32)

    residual_history = [1]
    for epoch in range(max_iter):
        

        ## ALS 1
        KR(KR_blocks,KR_blockdim,(Rc,Rb,KR_bc,min_rank3,min_rank2,rank))
        Q0,R0 = cp.linalg.qr(KR_bc)
        Qc = cp.ascontiguousarray(Qc)
        Qb = cp.ascontiguousarray(Qb)


        # Use cuTENSOR vs my implementation
        if cutensor_ttm:
            Y = ttm_rank2(Qb.T,Qc.T,X,skip=1)
        else:
            ttm_launcher(Qb,Qc,X,ttm1_first,ttm1_second,dim1,dim2,dim3,rank,ttm_kernels,skip=1)
            Y = ttm1_second
        Y0 = f_unfold(Y,mode=0)
        W = Y0@Q0

        # Case for short, fat matrices Rank>>dim1,dim2,dim3
        if R0.shape[0]!=R0.shape[1]:
            a = (linalg.solve_triangular(R0[:,0:W.T.shape[0]],W.T).T)@R0
        else:
            a = linalg.solve_triangular(R0,W.T).T
        Qa,Ra = cp.linalg.qr(a)
        Ra = Ra.astype(cp.float32)


        ##ALS 2
        KR(KR_blocks,KR_blockdim,(Rc,Ra,KR_ac,min_rank3,min_rank1,rank))
        Q1,R1 = cp.linalg.qr(KR_ac)
        Qc = cp.ascontiguousarray(Qc)
        Qa = cp.ascontiguousarray(Qa)
        if cutensor_ttm:
            Y = ttm_rank2(Qa.T,Qc.T,X,skip=2)
        else:
            ttm_launcher(Qa,Qc,X,ttm2_first,ttm2_second,dim1,dim2,dim3,rank,ttm_kernels,skip=2)
            Y = ttm2_second


        Y0 = f_unfold(Y,mode=1)
        W= Y0@Q1
        if R1.shape[0]!=R1.shape[1]:
            b = (linalg.solve_triangular(R1[:,0:W.T.shape[0]],W.T).T)@R1
        else:
            b = linalg.solve_triangular(R1,W.T).T
        Qb,Rb = cp.linalg.qr(b)
        Rb = Rb.astype(cp.float32)


        ##ALS 3
        KR(KR_blocks,KR_blockdim,(Rb,Ra,KR_ab,min_rank2,min_rank1,rank))
        Q2,R2 = cp.linalg.qr(KR_ab)
        Qa = cp.ascontiguousarray(Qa)
        Qb = cp.ascontiguousarray(Qb)
        if cutensor_ttm:
            Y = ttm_rank2(Qa.T,Qb.T,X,skip=3)
        else:
            ttm_launcher(Qa,Qb,X,ttm3_first,ttm3_second,dim1,dim2,dim3,rank,ttm_kernels,skip=3)
            Y = ttm3_second
        Y0 = f_unfold(Y,mode=2)
        W= Y0@Q2
        if R2.shape[0]!=R2.shape[1]:
            c = (linalg.solve_triangular(R2[:,0:W.T.shape[0]],W.T).T)@R2
        else:
            c = linalg.solve_triangular(R2,W.T).T
        Qc,Rc = cp.linalg.qr(c)
        Rc = Rc.astype(cp.float32)
        

        if verbose:
            res_a = cp.linalg.norm(khatri_rao(c, b)@(a.T) - f_unfold(X,mode=0).T,'fro')
            res_b = cp.linalg.norm(khatri_rao(c, a)@(b.T) - f_unfold(X,mode=1).T,'fro')
            res_c = cp.linalg.norm(khatri_rao(b, a)@(c.T) - f_unfold(X,mode=2).T,'fro')
            print("Epoch:", epoch, "Rel Error: ", max([res_a/res_a1,res_b/res_b1,res_c/res_c1]))
            residual_history.append( max([res_a/res_a1,res_b/res_b1,res_c/res_c1]).get())
            print("Epoch:", epoch, "| Loss (A):", res_a, "| Loss (B):", res_b, "| Loss (C):", res_c)
            if max([res_a/res_a1,res_b/res_b1,res_c/res_c1])< rel_tol:
                print(f"ALS converged in {epoch} iterations")
                break

    return a.T, b.T, c.T,residual_history