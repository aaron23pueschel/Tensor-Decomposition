
import numpy as np
from scipy import linalg


def f_unfold(tensor, mode=0,order='F'):
    """
    Returns an unfolded mode of a tensor.

    Parameters
    ----------
    tensor : numpy.array
        Order 3 tensor.
    mode : int, optional
        The mode unfolding of the tensor. Defaults to 0.
    order : str, optional
        Define Column or Row major ordering for the unfolding. Defaults to Fortran unfolding.

    Returns
    -------
    numpy.array
        The unfolded mode of the tensor.
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order=order)



def khatri_rao(a,b):
    """
    Returns the Khatri Rao product of two factor matrices. Verify the second dimensions are equal.

    Parameters
    ----------
    a : numpy.array
        First matrix.
    b : numpy.array
        Second matrix. Both matrices should have the same number of columns.

    Returns
    -------
    numpy.array
        The Khatri Rao product of the two matrices.
    """
    return (a[:,None] * b).reshape(-1,a.shape[1])



def ttm_rank2(Q1,Q2,X,skip=1):
    """
    Returns the second order tensor matrix product.

    Parameters
    ----------
    Q1 : numpy.array
        First factor matrix.
    Q2 : numpy.array
        Second factor matrix.
    X : numpy.array
        3D matrix.
    skip : int, optional
        Choose which dimension to skip. Defaults to 1.

    Returns
    -------
    numpy.array
        The tensor matrix product.

    """


    if skip == 1:
        X = np.einsum('jl,ilk->ijk', Q1.T,X)
        return np.einsum('kl,ijl->ijk', Q2.T, X)
    elif skip ==2:
        X = np.einsum('jh,hlk->jlk', Q1.T,X)
        return np.einsum('kl,ijl->ijk', Q2.T, X)
    elif skip==3:
        X = np.einsum('jh,hlk->jlk', Q1.T,X)
        return  np.einsum('jl,ilk->ijk', Q2.T,X)


def decompose_normal_eqns(tensor, factor_matrices = None,rank=10, max_iter=501, rel_tol = 10e-10,verbose=True):
    """
    Decompose a tensor using normal equations.

    Parameters
    ----------
    tensor : cupy.ndarray
        The tensor to decompose.
    factor_matrices : cupy.ndarray, optional
        Initial factor matrices for the decomposition. Defaults to None.
    rank : int, optional
        The rank of the decomposition. Defaults to 10.
    max_iter : int, optional
        The maximum number of iterations for the decomposition. Defaults to 501.
    rel_tol : float, optional
        The relative tolerance for convergence. Defaults to 1e-10.
    verbose : bool, optional
        Whether to print verbose output. Defaults to True.

    Returns
    -------
    cupy.ndarray
        The decomposed factor matrices.
    """
    if factor_matrices is not None:
        a = factor_matrices[0]
        b = factor_matrices[1]
        c = factor_matrices[2]
    else:
        a = np.random.random((tensor.shape[0],rank))
        b = np.random.random((tensor.shape[1],rank))
        c = np.random.random((tensor.shape[2],rank))
    
    for epoch in range(max_iter):
        # optimize a
        input_a = khatri_rao(b,c)
        target_a = f_unfold(tensor, mode=0,order='C').T
       
        a = np.linalg.solve(input_a.T@input_a, input_a.T@target_a).T
        
        # optimize b
        input_b = khatri_rao(a,c)
        target_b = f_unfold(tensor, mode=1,order='C').T
        b = np.linalg.solve(input_b.T@input_b, input_b.T@target_b).T

        # optimize c
        input_c = khatri_rao(a,b)
        target_c = f_unfold(tensor, mode=2,order='C').T
        c = np.linalg.solve(input_c.T@input_c, input_c.T@target_c).T

        res_a = np.linalg.norm(input_a.dot(a.T) - target_a,'fro')
        res_b = np.linalg.norm(input_b.dot(b.T) - target_b,'fro')
        res_c = np.linalg.norm(input_c.dot(c.T) - target_c,'fro')

        if verbose:
            print("Epoch:", epoch, "| Loss (A):", res_a, "| Loss (B):", res_b, "| Loss (C):", res_c)
            if max([res_a,res_b,res_c])< rel_tol:
                print(f"ALS converged in {epoch} iterations")
                break
    return a.T, b.T, c.T



def decompose_QR(X, factor_matrices = None,rank=10, max_iter=501, rel_tol = 10e-10,verbose=True):
    """
    Decompose tensor into CPD decomposition using QR

    Parameters
    ----------
    X : cupy.ndarray
        The tensor to decompose.
    factor_matrices : cupy.ndarray, optional
        Initial factor matrices for the decomposition. Defaults to None.
    rank : int, optional
        The rank of the decomposition. Defaults to 10.
    max_iter : int, optional
        The maximum number of iterations for the decomposition. Defaults to 501.
    rel_tol : float, optional
        The relative tolerance for convergence. Defaults to 1e-10.
    verbose : bool, optional
        Whether to print verbose output. Defaults to True.

    Returns
    -------
    cupy.ndarray
        The decomposed factor matrices.
    """
    if factor_matrices is not None:
        a = factor_matrices[0]
        b = factor_matrices[1]
        c = factor_matrices[2]
    else:
        a = np.random.random((X.shape[0],rank))
        b = np.random.random((X.shape[1],rank))
        c = np.random.random((X.shape[2],rank))
    Qa,Ra = np.linalg.qr(a)
    Qb,Rb = np.linalg.qr(b)
    Qc,Rc = np.linalg.qr(c)
    for epoch in range(max_iter):
        
        V0 = khatri_rao(Rc, Rb)
        Q0,R0 = np.linalg.qr(V0)
        Y = ttm_rank2(Qb,Qc,X)
        Y0 = f_unfold(Y,mode=0)
        W= Y0@Q0
        a = linalg.solve_triangular(R0,W.T).T
        Qa,Ra = np.linalg.qr(a)


        V1 = khatri_rao(Rc, Ra)
        Q1,R1 = np.linalg.qr(V1)
        Y = ttm_rank2(Qa,Qc,X,skip=2)
        Y0 = f_unfold(Y,mode=1)
        W= Y0@Q1
        b = linalg.solve_triangular(R1,W.T).T
        Qb,Rb = np.linalg.qr(b)

        V2 = khatri_rao(Rb, Ra)
        Q2,R2 = np.linalg.qr(V2)
        Y = ttm_rank2(Qa,Qb,X,skip=3)
        Y0 = f_unfold(Y,mode=2)
        W= Y0@Q2
        c = linalg.solve_triangular(R2,W.T).T
        Qc,Rc = np.linalg.qr(c)


        res_a = np.linalg.norm(khatri_rao(c, b)@(a.T) - f_unfold(X,mode=0,order='F').T,'fro')
        res_b = np.linalg.norm(khatri_rao(c, a)@(b.T) - f_unfold(X,mode=1,order='F').T,'fro')
        res_c = np.linalg.norm(khatri_rao(b, a)@(c.T) - f_unfold(X,mode=2,order='F').T,'fro')

        if verbose:
            print("Epoch:", epoch, "| Loss (A):", res_a, "| Loss (B):", res_b, "| Loss (C):", res_c)
        if max([res_a,res_b,res_c])< rel_tol:
            print(f"ALS converged in {epoch} iterations")
            break
    return a.T, b.T, c.T